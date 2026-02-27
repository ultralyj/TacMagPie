/**
 * @file mlx90393.h
 * @author Luo-Yijie (1951578@tongji.edu.cn)
 * @brief mlx90393 drive code (header)
 * @version 0.1
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

/**
 * Component header file.
 */
#ifndef __MLX90393_H__
#define __MLX90393_H__

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "main.h"
#include "stm32g0xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * Device specifics (registers, NVRAM, etc.)
 */
#include "mlx90393_defs.h"

/**
 * I2C Settings
 */
#define MLX90393_I2C_TIMEOUT_DEFAULT        (1000)  /*!< Timeout in milliseconds */

/**
 * My default MLX device settings - see mlx90393_init() - recommended settings from the DATA SHEET.
 */
#define MLX90393_COMM_MODE_DEFAULT  (MLX90393_COMM_MODE_I2C)
#define MLX90393_TCMP_EN_DEFAULT    (MLX90393_TCMP_EN_DISABLED)
#define MLX90393_HALLCONF_DEFAULT   (MLX90393_HALLCONF_C)
#define MLX90393_GAIN_SEL_DEFAULT   (MLX90393_GAIN_SEL_3)
#define MLX90393_RES_XYZ_DEFAULT    (MLX90393_RES_XYZ_1)
#define MLX90393_DIG_FILT_DEFAULT   (MLX90393_DIG_FILT_5)
#define MLX90393_OSR_DEFAULT        (MLX90393_OSR_1)
#define MLX90393_OFFSET_X_DEFAULT  (0)
#define MLX90393_OFFSET_Y_DEFAULT  (0)
#define MLX90393_OFFSET_Z_DEFAULT  (0)

/**
 * Data structs
 */
typedef struct {
        bool temperature;
        bool x_axis;
        bool y_axis;
        bool z_axis;
} mlx90393_metrics_selector_t;

/*
 * mlx90393_config_t
 *      int_gpio_num : Melexis INT DRDY Data Ready pin @rule -1 means not used to detect that a measurement is ready to be read.
 */
typedef struct {
        uint8_t manage_i2c_driver;
        I2C_HandleTypeDef* i2c_handle;
        uint8_t i2c_slave_addr;
        GPIO_TypeDef* int_gpio_port;
        uint16_t int_gpio_pin;

        mlx90393_metrics_selector_t mlx_metrics_selector;

        uint16_t mlx_sens_tc_lt;
        uint16_t mlx_sens_tc_ht;
        uint16_t mlx_tref;

        mlx90393_comm_mode_t mlx_comm_mode;
        mlx90393_tcmp_en_t mlx_tcmp_en;
        mlx90393_hallconf_t mlx_hallconf;
        mlx90393_gain_sel_t mlx_gain_sel;
        mlx90393_osr_t mlx_osr;
        mlx90393_dig_filt_t mlx_dig_filt;
        mlx90393_res_xyz_t mlx_res_x, mlx_res_y, mlx_res_z;
        uint16_t mlx_offset_x, mlx_offset_y, mlx_offset_z;
        uint16_t conv_time;
} mlx90393_config_t;

#define MLX90393_CONFIG_DEFAULT() { \
    .i2c_handle = NULL, \
    .i2c_slave_addr = 0x0C, \
    .int_gpio_port = NULL, \
    .int_gpio_pin = 0, \
    .mlx_metrics_selector = { \
            .x_axis = true, \
            .y_axis = true, \
            .z_axis = true, \
            .temperature = true \
    } \
};

typedef struct {
        uint16_t t;
        uint16_t x;
        uint16_t y;
        uint16_t z;
} mlx90393_data_raw_t;

typedef struct {
        uint16_t t_raw;
        uint16_t x_raw;
        uint16_t y_raw;
        uint16_t z_raw;
        float t;
        float x;
        float y;
        float z;
} mlx90393_data_t;

/**
 * Function declarations
 */
HAL_StatusTypeDef mlx90393_init(mlx90393_config_t* param_ptr_config);
HAL_StatusTypeDef mlx90393_deinit(const mlx90393_config_t* param_ptr_config);

HAL_StatusTypeDef mlx90393_log_device_parameters(const mlx90393_config_t* param_ptr_config);

HAL_StatusTypeDef mlx90393_cmd_reset(const mlx90393_config_t* param_ptr_config);
HAL_StatusTypeDef mlx90393_cmd_exit(const mlx90393_config_t* param_ptr_config);

HAL_StatusTypeDef mlx90393_get_comm_mode(const mlx90393_config_t* param_ptr_config, mlx90393_comm_mode_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_tcmp_en(const mlx90393_config_t* param_ptr_config, mlx90393_tcmp_en_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_hallconf(const mlx90393_config_t* param_ptr_config, mlx90393_hallconf_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_gain_sel(const mlx90393_config_t* param_ptr_config, mlx90393_gain_sel_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_z_series(const mlx90393_config_t* param_ptr_config, mlx90393_z_series_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_bist(const mlx90393_config_t* param_ptr_config, mlx90393_bist_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_ext_trig(const mlx90393_config_t* param_ptr_config, mlx90393_ext_trig_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_trig_int_sel(const mlx90393_config_t* param_ptr_config, mlx90393_trig_int_sel_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_osr(const mlx90393_config_t* param_ptr_config, mlx90393_osr_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_dig_filt(const mlx90393_config_t* param_ptr_config, mlx90393_dig_filt_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_res_xyz(const mlx90393_config_t* param_ptr_config, mlx90393_res_xyz_t* param_x, mlx90393_res_xyz_t* param_y,
                                   mlx90393_res_xyz_t* param_z);
HAL_StatusTypeDef mlx90393_get_sens_tc_lt(const mlx90393_config_t* param_ptr_config, uint8_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_sens_tc_ht(const mlx90393_config_t* param_ptr_config, uint8_t* param_ptr_data);
HAL_StatusTypeDef mlx90393_get_offset_xyz(const mlx90393_config_t* param_ptr_config, uint16_t* param_x, uint16_t* param_y, uint16_t* param_z);
HAL_StatusTypeDef mlx90393_get_tref(const mlx90393_config_t* param_ptr_config, uint16_t* param_ptr_data);

HAL_StatusTypeDef mlx90393_set_comm_mode(mlx90393_config_t* param_ptr_config, mlx90393_comm_mode_t param_data);
HAL_StatusTypeDef mlx90393_set_tcmp_en(mlx90393_config_t* param_ptr_config, mlx90393_tcmp_en_t param_data);
HAL_StatusTypeDef mlx90393_set_hallconf(mlx90393_config_t* param_ptr_config, mlx90393_hallconf_t param_data);
HAL_StatusTypeDef mlx90393_set_gain_sel(mlx90393_config_t* param_ptr_config, mlx90393_gain_sel_t param_data);
HAL_StatusTypeDef mlx90393_set_osr(mlx90393_config_t* param_ptr_config, mlx90393_osr_t param_data);
HAL_StatusTypeDef mlx90393_set_dig_filt(mlx90393_config_t* param_ptr_config, mlx90393_dig_filt_t param_data);
HAL_StatusTypeDef mlx90393_set_res_xyz(mlx90393_config_t* param_ptr_config, mlx90393_res_xyz_t param_res_x, mlx90393_res_xyz_t param_res_y,
                                   mlx90393_res_xyz_t param_res_z);
HAL_StatusTypeDef mlx90393_set_offset_xyz(mlx90393_config_t* param_ptr_config, uint16_t param_offset_x, uint16_t param_offset_y,
                                      uint16_t param_offset_z);
HAL_StatusTypeDef mlx90393_cmd_start_measurement(const mlx90393_config_t* param_ptr_config);
HAL_StatusTypeDef mlx90393_cmd_read_measurement(const mlx90393_config_t* param_ptr_config, mlx90393_data_t* param_ptr_data);

#ifdef __cplusplus
}
#endif

#endif /* __MLX90393_H__ */