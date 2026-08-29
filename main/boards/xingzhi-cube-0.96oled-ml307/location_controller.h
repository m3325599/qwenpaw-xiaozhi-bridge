#ifndef LOCATION_CONTROLLER_H
#define LOCATION_CONTROLLER_H

#include <sdkconfig.h>

#ifndef CONFIG_LOCATION_AMAP_KEY
#define CONFIG_LOCATION_AMAP_KEY ""
#endif

#include "mcp_server.h"
#include "board.h"
#include "audio_codec.h"
#include "application.h"
#include <at_modem.h>
#include <at_uart.h>
#include <string>
#include <variant>
#include <vector>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <cJSON.h>

#define TAG "LocationController"

class LocationController {
public:
    LocationController() {
        bool use_amap = (CONFIG_LOCATION_GEOCODING_PROVIDER == 1) ||
                        (std::string(CONFIG_LOCATION_AMAP_KEY).length() > 0);
        if (use_amap) {
            ESP_LOGI(TAG, "Location geocoding provider: Amap (key=%.8s...)", CONFIG_LOCATION_AMAP_KEY);
        } else {
            ESP_LOGI(TAG, "Location geocoding provider: BigDataCloud");
        }
        auto& mcp_server = McpServer::GetInstance();

        // ML307R-DL 模块不支持 GNSS 卫星定位
        mcp_server.AddTool("self.location.get_gnss",
            "This device uses ML307R-DL module which does not support GNSS satellite positioning.\n"
            "Please use self.location.get_lbs for cellular positioning.",
            PropertyList(),
            [this](const PropertyList& properties) -> ReturnValue {
                ESP_LOGI(TAG, "GNSS requested but ML307R-DL does not support GNSS");
                return std::string("{\"error\":\"GNSS satellite positioning is not supported on this device (ML307R-DL module). Please use cellular LBS positioning instead.\"}");
            });

        mcp_server.AddTool("self.location.get_lbs",
            "Get current location using cellular base station positioning (LBS).\n"
            "Works without GNSS antenna, only needs SIM card and cellular network.\n"
            "Accuracy typically 50-500 meters. Response time is usually 3-15 seconds.\n"
            "Returns JSON with type, latitude, longitude, accuracy, address and a Chinese summary.\n"
            "After receiving the result, you MUST briefly report the location to the user in Chinese, including the address and approximate accuracy.",
            PropertyList(),
            [this](const PropertyList& properties) -> ReturnValue {
                return HandleGetLbs();
            });

        mcp_server.AddTool("self.location.get_current",
            "Get current location using cellular LBS positioning.\n"
            "Fast and reliable. Does not require GNSS antenna. Response time is usually 3-15 seconds.\n"
            "Returns JSON with type, latitude, longitude, accuracy, address and a Chinese summary.\n"
            "After receiving the result, you MUST briefly report the location to the user in Chinese, including the address and approximate accuracy.",
            PropertyList(),
            [this](const PropertyList& properties) -> ReturnValue {
                return HandleGetLbs();
            });
    }

    ~LocationController() {}

private:
    bool lbs_configured_ = false;

    void SetDisplayMessage(const std::string& message) {
        auto* display = Board::GetInstance().GetDisplay();
        if (display) {
            display->SetChatMessage("system", message.c_str());
        }
    }

    AtModem* GetModem() {
        auto* network = Board::GetInstance().GetNetwork();
        return dynamic_cast<AtModem*>(network);
    }

    void PauseAudioOutput() {
        auto& app = Application::GetInstance();
        ESP_LOGI(TAG, "PauseAudioOutput: state=%d", (int)app.GetDeviceState());
        app.AbortSpeaking(kAbortReasonNone);
        // 暂停音频输出并等待 UHCI 资源释放；ML307 模组定位不需要长时间占用，3 秒通常足够
        ESP_LOGI(TAG, "Waiting 3s for UHCI to become free...");
        vTaskDelay(pdMS_TO_TICKS(3000));
        auto* audio_codec = Board::GetInstance().GetAudioCodec();
        if (audio_codec) {
            ESP_LOGI(TAG, "Disabling I2S audio output...");
            audio_codec->EnableOutput(false);
            vTaskDelay(pdMS_TO_TICKS(200));
        }
    }

    void ResumeAudioOutput() {
        auto* audio_codec = Board::GetInstance().GetAudioCodec();
        if (audio_codec) {
            ESP_LOGI(TAG, "Enabling I2S audio output...");
            audio_codec->EnableOutput(true);
        }
    }

    bool SendAtCommand(const std::string& cmd, int timeout_ms = 3000, int max_retries = 5, int delay_ms = 1000) {
        auto* modem = GetModem();
        if (!modem) {
            ESP_LOGE(TAG, "SendAtCommand: modem not available");
            return false;
        }
        auto at_uart = modem->GetAtUart();
        ESP_LOGI(TAG, "SendAtCommand: [%s] (timeout=%d, retries=%d)", cmd.c_str(), timeout_ms, max_retries);
        for (int retry = 0; retry < max_retries; retry++) {
            bool ok = at_uart->SendCommand(cmd, timeout_ms);
            ESP_LOGI(TAG, "  Attempt %d/%d: %s", retry + 1, max_retries, ok ? "OK" : "FAIL");
            if (ok) return true;
            if (retry < max_retries - 1) {
                ESP_LOGW(TAG, "  Retry %d/%d for: %s", retry + 1, max_retries, cmd.c_str());
                vTaskDelay(pdMS_TO_TICKS(delay_ms));
            }
        }
        ESP_LOGE(TAG, "SendAtCommand FAILED: %s", cmd.c_str());
        return false;
    }

    bool ConfigureLbs() {
        if (lbs_configured_) {
            ESP_LOGI(TAG, "LBS already configured");
            return true;
        }
        ESP_LOGI(TAG, "Configuring LBS (OneOS, method=40)...");
        if (!SendAtCommand("AT+MLBSCFG=\"method\",40")) {
            ESP_LOGE(TAG, "Failed to set LBS method");
            return false;
        }
        SendAtCommand("AT+MLBSCFG=\"nearbtsen\",1");
        lbs_configured_ = true;
        ESP_LOGI(TAG, "LBS configured");
        return true;
    }

    std::string ReverseGeocode(const std::string& latitude, const std::string& longitude) {
        // 只要配置了高德 key 就优先使用高德；避免 sdkconfig 缓存导致 provider 未生效
        bool use_amap = (CONFIG_LOCATION_GEOCODING_PROVIDER == 1) ||
                        (std::string(CONFIG_LOCATION_AMAP_KEY).length() > 0);
        if (use_amap) {
            std::string address = ReverseGeocodeAmap(latitude, longitude);
            if (!address.empty()) {
                return address;
            }
            ESP_LOGW(TAG, "Amap geocoding failed, fallback to BigDataCloud");
        }
        return ReverseGeocodeBigDataCloud(latitude, longitude);
    }

    std::string ReverseGeocodeBigDataCloud(const std::string& latitude, const std::string& longitude) {
        std::string url = "https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=" +
                          latitude + "&longitude=" + longitude + "&localityLanguage=zh";
        ESP_LOGI(TAG, "Reverse geocoding (BigDataCloud): %s", url.c_str());

        auto network = Board::GetInstance().GetNetwork();
        auto http = network->CreateHttp(0);
        if (!http->Open("GET", url)) {
            ESP_LOGE(TAG, "Failed to open geocoding URL");
            return "";
        }

        if (http->GetStatusCode() != 200) {
            ESP_LOGE(TAG, "Geocoding failed, status: %d", http->GetStatusCode());
            http->Close();
            return "";
        }

        std::string response = http->ReadAll();
        http->Close();

        cJSON* root = cJSON_Parse(response.c_str());
        if (!root) {
            ESP_LOGE(TAG, "Failed to parse geocoding response");
            return "";
        }

        std::string address;
        auto append_field = [&](const char* key) {
            cJSON* item = cJSON_GetObjectItem(root, key);
            if (cJSON_IsString(item) && strlen(item->valuestring) > 0) {
                if (!address.empty()) {
                    address += " ";
                }
                address += item->valuestring;
            }
        };

        append_field("countryName");
        append_field("principalSubdivision");
        append_field("city");
        append_field("locality");

        cJSON_Delete(root);
        ESP_LOGI(TAG, "Reverse geocoding result: %s", address.c_str());
        return address;
    }

    std::string ReverseGeocodeAmap(const std::string& latitude, const std::string& longitude) {
        std::string url = "https://restapi.amap.com/v3/geocode/regeo?key=" +
                          std::string(CONFIG_LOCATION_AMAP_KEY) +
                          "&location=" + longitude + "," + latitude +
                          "&extensions=all&coordsys=wgs84";
        ESP_LOGI(TAG, "Reverse geocoding (Amap): %s", url.c_str());

        auto network = Board::GetInstance().GetNetwork();
        auto http = network->CreateHttp(0);
        if (!http->Open("GET", url)) {
            ESP_LOGE(TAG, "Failed to open Amap geocoding URL");
            return "";
        }

        if (http->GetStatusCode() != 200) {
            ESP_LOGE(TAG, "Amap geocoding failed, status: %d", http->GetStatusCode());
            http->Close();
            return "";
        }

        std::string response = http->ReadAll();
        http->Close();

        cJSON* root = cJSON_Parse(response.c_str());
        if (!root) {
            ESP_LOGE(TAG, "Failed to parse Amap geocoding response");
            return "";
        }

        std::string address;
        cJSON* status = cJSON_GetObjectItem(root, "status");
        if (cJSON_IsString(status) && strcmp(status->valuestring, "1") == 0) {
            cJSON* regeocode = cJSON_GetObjectItem(root, "regeocode");
            if (cJSON_IsObject(regeocode)) {
                cJSON* formatted = cJSON_GetObjectItem(regeocode, "formatted_address");
                if (cJSON_IsString(formatted) && strlen(formatted->valuestring) > 0) {
                    address = formatted->valuestring;
                }

                cJSON* component = cJSON_GetObjectItem(regeocode, "addressComponent");
                if (cJSON_IsObject(component)) {
                    auto append_string = [&](const char* key) {
                        cJSON* item = cJSON_GetObjectItem(component, key);
                        if (cJSON_IsString(item) && strlen(item->valuestring) > 0) {
                            if (!address.empty()) address += " ";
                            address += item->valuestring;
                        }
                    };

                    auto append_object_name = [&](const char* key) {
                        cJSON* item = cJSON_GetObjectItem(component, key);
                        if (cJSON_IsObject(item)) {
                            cJSON* name = cJSON_GetObjectItem(item, "name");
                            if (cJSON_IsString(name) && strlen(name->valuestring) > 0) {
                                if (!address.empty()) address += " ";
                                address += name->valuestring;
                            }
                        }
                    };

                    auto append_street_number = [&]() {
                        cJSON* item = cJSON_GetObjectItem(component, "streetNumber");
                        if (cJSON_IsObject(item)) {
                            cJSON* street = cJSON_GetObjectItem(item, "street");
                            cJSON* number = cJSON_GetObjectItem(item, "number");
                            std::string street_number;
                            if (cJSON_IsString(street) && strlen(street->valuestring) > 0) {
                                street_number += street->valuestring;
                            }
                            if (cJSON_IsString(number) && strlen(number->valuestring) > 0) {
                                street_number += number->valuestring;
                            }
                            if (!street_number.empty()) {
                                if (!address.empty()) address += " ";
                                address += street_number;
                            }
                        }
                    };

                    if (address.empty()) {
                        // 没有 formatted_address，从 addressComponent 拼装完整地址
                        append_string("country");
                        append_string("province");
                        append_string("city");
                        append_string("district");
                        append_string("township");
                    }

                    // 追加更详细的子地址信息
                    append_object_name("neighborhood");   // 小区/社区
                    append_object_name("building");       // 建筑物名称
                    append_street_number();                // 街道+门牌号

                    if (address.empty()) {
                        append_string("street");
                        append_string("streetNumber");
                    }
                }
            }
        } else {
            cJSON* info = cJSON_GetObjectItem(root, "info");
            ESP_LOGE(TAG, "Amap geocoding error: %s", cJSON_IsString(info) ? info->valuestring : "unknown");
        }

        cJSON_Delete(root);
        ESP_LOGI(TAG, "Reverse geocoding result: %s", address.c_str());
        return address;
    }

    ReturnValue HandleGetLbs() {
        ESP_LOGI(TAG, "=== HandleGetLbs START ===");
        auto* modem = GetModem();
        if (!modem) {
            SetDisplayMessage("定位失败：模组不可用");
            return std::string("{\"error\":\"Modem not available\"}");
        }
        auto at_uart = modem->GetAtUart();

        PauseAudioOutput();
        SetDisplayMessage("正在定位…");

        if (!ConfigureLbs()) {
            SetDisplayMessage("定位失败：LBS 配置失败");
            ResumeAudioOutput();
            return std::string("{\"error\":\"Failed to configure LBS\"}");
        }

        // 最多重试 5 次 LBS 定位
        const int max_attempts = 5;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            ESP_LOGI(TAG, "LBS attempt %d/%d", attempt + 1, max_attempts);
            SetDisplayMessage("正在获取基站信息…（" + std::to_string(attempt + 1) + "/" + std::to_string(max_attempts) + "）");

            SemaphoreHandle_t semaphore = xSemaphoreCreateBinary();
            if (!semaphore) {
                SetDisplayMessage("定位失败：内部错误");
                ResumeAudioOutput();
                return std::string("{\"error\":\"Failed to create semaphore\"}");
            }

            int status_code = 0;
            std::string longitude, latitude, accuracy;
            bool got_location = false;
            bool got_status = false;

            ESP_LOGI(TAG, "Registering URC callback for MLBSLOC...");
            auto callback_it = at_uart->RegisterUrcCallback(
                [&](const std::string& command, const std::vector<AtArgumentValue>& arguments) {
                    ESP_LOGI(TAG, "URC fired: command=%s, args=%d", command.c_str(), (int)arguments.size());
                    if (command != "MLBSLOC") return;
                    if (arguments.empty()) {
                        ESP_LOGW(TAG, "URC MLBSLOC: no arguments");
                        return;
                    }

                    status_code = arguments[0].int_value;
                    got_status = true;
                    ESP_LOGI(TAG, "URC MLBSLOC status code: %d", status_code);

                    // 收到错误状态码立即放行，不要等待 15 秒超时
                    if (status_code != 100) {
                        ESP_LOGW(TAG, "URC MLBSLOC error status: %d", status_code);
                        xSemaphoreGive(semaphore);
                        return;
                    }

                    if (arguments.size() >= 3) {
                        longitude = arguments[1].type == AtArgumentValue::Type::String ?
                                    arguments[1].string_value : std::to_string(arguments[1].double_value);
                        latitude = arguments[2].type == AtArgumentValue::Type::String ?
                                   arguments[2].string_value : std::to_string(arguments[2].double_value);
                        if (arguments.size() >= 4) {
                            accuracy = arguments[3].type == AtArgumentValue::Type::Int ?
                                       std::to_string(arguments[3].int_value) :
                                       (arguments[3].type == AtArgumentValue::Type::String ?
                                        arguments[3].string_value : std::to_string(arguments[3].double_value));
                        } else {
                            accuracy = "500";
                        }
                        ESP_LOGI(TAG, "URC MLBSLOC location: lon=%s, lat=%s, acc=%s",
                                 longitude.c_str(), latitude.c_str(), accuracy.c_str());
                        got_location = true;
                        xSemaphoreGive(semaphore);
                    } else {
                        ESP_LOGW(TAG, "URC MLBSLOC: success status but no coordinates");
                        // 只收到 100 状态码但没有坐标，继续等待完整 URC
                    }
                });

            ESP_LOGI(TAG, "Sending AT+MLBSLOC...");
            if (!SendAtCommand("AT+MLBSLOC", 30000, 1)) {
                at_uart->UnregisterUrcCallback(callback_it);
                vSemaphoreDelete(semaphore);
                continue;
            }

            // 检查同步响应：部分模组会在命令同步响应里直接返回 +MLBSLOC: 结果
            std::string response = at_uart->GetResponse();
            ESP_LOGI(TAG, "LBS GetResponse: [%s]", response.c_str());
            if (!response.empty() && response.find("+MLBSLOC:") != std::string::npos) {
                if (ParseLbsResponse(response, status_code, longitude, latitude, accuracy)) {
                    got_status = true;
                    if (status_code == 100) {
                        got_location = true;
                        ESP_LOGI(TAG, "LBS location obtained from synchronous response");
                    }
                }
            }

            // 同步响应已拿到坐标，无需等待 URC
            if (got_location) {
                at_uart->UnregisterUrcCallback(callback_it);
                vSemaphoreDelete(semaphore);
                break;
            }

            // 同步响应只有错误状态码，不等待 URC，直接按状态码处理
            if (got_status && status_code != 100) {
                at_uart->UnregisterUrcCallback(callback_it);
                vSemaphoreDelete(semaphore);
                if (status_code == 124 && attempt < max_attempts - 1) {
                    int delay_ms = 5000 + attempt * 5000;  // 5s, 10s, 15s, 20s
                    ESP_LOGW(TAG, "LBS status 124 on attempt %d, retrying after %d ms...", attempt + 1, delay_ms);
                    SetDisplayMessage("基站定位忙，" + std::to_string(delay_ms / 1000) + "秒后重试…");
                    vTaskDelay(pdMS_TO_TICKS(delay_ms));
                    continue;
                } else {
                    ResumeAudioOutput();
                    return std::string("{\"error\":\"LBS failed with status code " + std::to_string(status_code) + "\"}");
                }
            }

            // 等待 URC 回调，最多 20 秒（ML307 模组首次 LBS 定位可能较慢）
            if (!got_location) {
                ESP_LOGI(TAG, "Waiting for URC callback (max 20s)...");
                if (xSemaphoreTake(semaphore, pdMS_TO_TICKS(20000)) == pdTRUE) {
                    ESP_LOGI(TAG, "URC callback received location");
                } else {
                    ESP_LOGW(TAG, "URC callback timeout on attempt %d", attempt + 1);
                }
            }

            at_uart->UnregisterUrcCallback(callback_it);
            vSemaphoreDelete(semaphore);

            if (got_location && status_code == 100) {
                ResumeAudioOutput();
                SetDisplayMessage("正在解析位置…");
                std::string address = ReverseGeocode(latitude, longitude);

                // 生成中文摘要，方便 LLM 直接播报
                std::string summary = "当前定位在" + address + "附近，定位精度约" + accuracy + "米。";

                // 同步在屏幕上显示定位结果，避免只显示工具调用名
                auto display = Board::GetInstance().GetDisplay();
                if (display) {
                    display->SetChatMessage("system", (address + "\n精度约" + accuracy + "米").c_str());
                }
                cJSON* json = cJSON_CreateObject();
                cJSON_AddStringToObject(json, "type", "lbs");
                cJSON_AddNumberToObject(json, "latitude", std::stod(latitude));
                cJSON_AddNumberToObject(json, "longitude", std::stod(longitude));
                cJSON_AddNumberToObject(json, "accuracy", std::stod(accuracy));
                cJSON_AddStringToObject(json, "address", address.c_str());
                cJSON_AddStringToObject(json, "summary", summary.c_str());
                char* str = cJSON_PrintUnformatted(json);
                std::string result(str);
                cJSON_free(str);
                cJSON_Delete(json);
                ESP_LOGI(TAG, "=== HandleGetLbs END: %s", result.c_str());
                return result;
            }

            if (got_status && status_code != 100) {
                if (status_code == 124 && attempt < max_attempts - 1) {
                    int delay_ms = 5000 + attempt * 5000;
                    SetDisplayMessage("基站定位忙，" + std::to_string(delay_ms / 1000) + "秒后重试…");
                    ESP_LOGW(TAG, "LBS status 124 on attempt %d, retrying after %d ms...", attempt + 1, delay_ms);
                    vTaskDelay(pdMS_TO_TICKS(delay_ms));
                } else {
                    // 已经收到明确错误状态，不需要再重试
                    SetDisplayMessage("定位失败：错误码 " + std::to_string(status_code));
                    ResumeAudioOutput();
                    return std::string("{\"error\":\"LBS failed with status code " + std::to_string(status_code) + "\"}");
                }
            }

            // 等待一会再重试
            if (!got_location && attempt < max_attempts - 1) {
                SetDisplayMessage("定位未成功，准备重试…");
                ESP_LOGI(TAG, "Waiting 3s before retry...");
                vTaskDelay(pdMS_TO_TICKS(3000));
            }
        }

        SetDisplayMessage("定位失败：网络未返回位置信息");
        ResumeAudioOutput();
        return std::string("{\"error\":\"LBS timeout: no location response from network after retries\"}");
    }

    // 解析 LBS 响应，返回 true 表示成功解析到经纬度坐标
    bool ParseLbsResponse(const std::string& response, int& status_code,
                          std::string& longitude, std::string& latitude, std::string& accuracy) {
        size_t pos = response.find("+MLBSLOC:");
        if (pos == std::string::npos) return false;
        std::string data = response.substr(pos + 9);
        while (!data.empty() && data[0] == ' ') data = data.substr(1);
        auto parts = SplitByComma(data);
        if (parts.empty()) return false;
        try {
            status_code = std::stoi(parts[0]);
            if (parts.size() >= 3) {
                longitude = parts[1];
                latitude = parts[2];
                accuracy = parts.size() >= 4 ? parts[3] : "500";
                return true;
            }
        } catch (...) {
            return false;
        }
        return false;
    }

    static std::vector<std::string> SplitByComma(const std::string& str) {
        std::vector<std::string> parts;
        std::string part;
        for (char c : str) {
            if (c == ',') {
                parts.push_back(part);
                part.clear();
            } else {
                part += c;
            }
        }
        if (!part.empty()) parts.push_back(part);
        return parts;
    }
};

#undef TAG

#endif // LOCATION_CONTROLLER_H