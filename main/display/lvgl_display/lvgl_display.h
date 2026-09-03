#ifndef LVGL_DISPLAY_H
#define LVGL_DISPLAY_H

#include "display.h"
#include "lvgl_image.h"

#include <lvgl.h>
#include <esp_timer.h>
#include <esp_log.h>
#include <esp_pm.h>

#include <string>
#include <chrono>

class LvglDisplay : public Display {
public:
    LvglDisplay();
    virtual ~LvglDisplay();

    virtual void SetStatus(const char* status);
    virtual void ShowNotification(const char* notification, int duration_ms = 3000);
    virtual void ShowNotification(const std::string &notification, int duration_ms = 3000);
    virtual void SetPreviewImage(std::unique_ptr<LvglImage> image);
    virtual void UpdateStatusBar(bool update_all = false);
    virtual void SetPowerSaveMode(bool on);
    virtual bool SnapshotToJpeg(std::string& jpeg_data, int quality = 80);

protected:
    esp_pm_lock_handle_t pm_lock_ = nullptr;
    lv_display_t *display_ = nullptr;

    lv_obj_t *network_label_ = nullptr;
    lv_obj_t *status_label_ = nullptr;
    lv_obj_t *notification_label_ = nullptr;
    lv_obj_t *mute_label_ = nullptr;
    lv_obj_t *battery_label_ = nullptr;
    lv_obj_t* low_battery_popup_ = nullptr;
    lv_obj_t* low_battery_label_ = nullptr;
    
    const char* battery_icon_ = nullptr;
    const char* network_icon_ = nullptr;
    bool muted_ = false;

    std::chrono::system_clock::time_point last_status_update_time_;
    esp_timer_handle_t notification_timer_ = nullptr;

    friend class DisplayLockGuard;
    virtual bool Lock(int timeout_ms = 0) = 0;
    virtual void Unlock() = 0;
};

#if CONFIG_USE_FULLSCREEN_TEXT_SCROLL
// Exec callback for the vertical scroll animation. Wraps lv_obj_set_scroll_y so
// the third (anim_enable) argument stays a valid LV_ANIM_OFF instead of relying
// on a function-pointer cast that would pass an undefined value.
static void lvgl_fullscreen_scroll_anim_cb(void* var, int32_t value) {
    lv_obj_set_scroll_y((lv_obj_t*)var, value, LV_ANIM_OFF);
}

// Starts a vertical up/down loop scroll for a wrapped label hosted inside a
// vertically scrollable container. The label must use LV_LABEL_LONG_WRAP with
// a fixed width and the container must have LV_DIR_VER scroll enabled.
static inline void lvgl_start_fullscreen_scroll(lv_obj_t* scroll_obj, lv_obj_t* label) {
    lv_obj_update_layout(scroll_obj);
    lv_obj_update_layout(label);
    lv_coord_t viewport_h = lv_obj_get_content_height(scroll_obj);
    lv_coord_t label_h = lv_obj_get_height(label);
    lv_coord_t max_scroll = label_h - viewport_h;

    // Stop any previous scroll animation and reset to the top
    lv_anim_delete(scroll_obj, lvgl_fullscreen_scroll_anim_cb);
    lv_obj_set_scroll_y(scroll_obj, 0, LV_ANIM_OFF);

    if (max_scroll <= 0) {
        return;
    }

    // Scroll at a constant reading speed, with a pause at both ends
    uint32_t duration = (uint32_t)max_scroll * 40;
    if (duration < 3000) duration = 3000;
    if (duration > 15000) duration = 15000;

    lv_anim_t a;
    lv_anim_init(&a);
    lv_anim_set_var(&a, scroll_obj);
    lv_anim_set_exec_cb(&a, lvgl_fullscreen_scroll_anim_cb);
    lv_anim_set_values(&a, 0, max_scroll);
    lv_anim_set_duration(&a, duration);
    lv_anim_set_playback_delay(&a, 1200);
    lv_anim_set_playback_time(&a, duration);
    lv_anim_set_repeat_delay(&a, 1200);
    lv_anim_set_repeat_count(&a, LV_ANIM_REPEAT_INFINITE);
    lv_anim_start(&a);
}
#endif


#endif
