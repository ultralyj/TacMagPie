/**
 * @file mlx90393.c
 * @author Luo-Yijie (1951578@tongji.edu.cn)
 * @brief mlx90393 drive code
 * @version 0.2
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "mlx90393.h"
#include <stdio.h>
#include "log.h"
static float _sensitivity_xy[MLX90393_GAIN_SEL_MAX][MLX90393_RES_XYZ_MAX] =
    {
        {0.751, 1.502, 3.004, 6.009},
        {0.601, 1.202, 2.403, 4.840},
        {0.451, 0.901, 1.803, 3.605},
        {0.376, 0.751, 1.502, 3.004},
        {0.300, 0.601, 1.202, 2.403},
        {0.250, 0.501, 1.001, 2.003},
        {0.200, 0.401, 0.801, 1.602},
        {0.150, 0.300, 0.601, 1.202},
};
static float _sensitivity_z[MLX90393_GAIN_SEL_MAX][MLX90393_RES_XYZ_MAX] =
    {
        {1.210, 2.420, 4.840, 9.680},
        {0.968, 1.936, 3.872, 7.744},
        {0.726, 1.452, 2.904, 5.808},
        {0.605, 1.210, 2.420, 4.840},
        {0.484, 0.968, 1.936, 3.872},
        {0.403, 0.807, 1.613, 3.227},
        {0.323, 0.645, 1.291, 2.581},
        {0.242, 0.484, 0.968, 1.936},
};

static HAL_StatusTypeDef _log_status_byte(uint8_t param_status) {
    LOG_Debug("  BURST_MODE_BIT?:              %u", (param_status & MLX90393_STATUS_BURST_MODE_BITMASK) != 0);
    LOG_Debug("  WAKE_ON_CHANGE_BIT?:          %u", (param_status & MLX90393_STATUS_WAKE_ON_CHANGE_MODE_BITMASK) != 0);
    LOG_Debug("  SINGLE_MEASUREMENT_MODE_BIT?: %u", (param_status & MLX90393_STATUS_SINGLE_MEASUREMENT_MODE_BITMASK) != 0);
    LOG_Debug("  ERROR_BIT?:                   %u", (param_status & MLX90393_STATUS_ERROR_BITMASK) != 0);
    LOG_Debug("  SED_BIT?:                     %u", (param_status & MLX90393_STATUS_SED_BITMASK) != 0);
    LOG_Debug("  RESET_BIT?:                   %u", (param_status & MLX90393_STATUS_RESET_BITMASK) != 0);
    LOG_Debug("  D1_BIT?:                      %u", (param_status & MLX90393_STATUS_D1_BITMASK) != 0);
    LOG_Debug("  D0_BIT?:                      %u", (param_status & MLX90393_STATUS_D0_BITMASK) != 0);
    return HAL_OK;
}

/*********************************************************************************
 * _send_cmd()
 *
 *  A command with no extra data to be sent and no extra data to be received
 *********************************************************************************/
static HAL_StatusTypeDef _send_cmd(const mlx90393_config_t *param_ptr_config, mlx90393_command_t param_command,
                                   mlx90393_status_byte_t *param_ptr_status_byte)
{
    HAL_StatusTypeDef ret = HAL_OK;
    uint8_t tx_data[1] = {param_command};
    uint8_t rx_data[1] = {0};

    // Send command
    ret = HAL_I2C_Master_Transmit(param_ptr_config->i2c_handle, param_ptr_config->i2c_slave_addr << 1, tx_data, 1, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK)
    {
        return ret;
    }

    // Receive status byte
    ret = HAL_I2C_Master_Receive(param_ptr_config->i2c_handle, param_ptr_config->i2c_slave_addr << 1, rx_data, 1, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK)
    {
        return ret;
    }

    *param_ptr_status_byte = rx_data[0];

    // Process STATUS BYTE
    // _log_status_byte(*param_ptr_status_byte);

    // Check status byte: error bit
    if ((*param_ptr_status_byte & MLX90393_STATUS_ERROR_BITMASK) != 0)
    {
        return HAL_ERROR;
    }

    return HAL_OK;
}

/*********************************************************************************
 * _read_register()
 *
 *  Read out the content of one specific address of the volatile RAM
 *********************************************************************************/
static HAL_StatusTypeDef _read_register(const mlx90393_config_t *param_ptr_config, mlx90393_reg_t param_reg,
                                        mlx90393_status_byte_t *param_ptr_status_byte, uint16_t *param_ptr_data)
{
    HAL_StatusTypeDef ret = HAL_OK;
    uint8_t tx_data[2] = {MLX90393_CMD_READ_REGISTER, param_reg << 2};
    uint8_t rx_data[3] = {0};

    // Send read register command
    ret = HAL_I2C_Master_Transmit(param_ptr_config->i2c_handle, param_ptr_config->i2c_slave_addr << 1, tx_data, 2, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK)
    {
        return ret;
    }

    // Receive status byte and register data
    ret = HAL_I2C_Master_Receive(param_ptr_config->i2c_handle, param_ptr_config->i2c_slave_addr << 1, rx_data, 3, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK)
    {
        return ret;
    }

    *param_ptr_status_byte = rx_data[0];
    *param_ptr_data = ((uint16_t)rx_data[1] << 8) | rx_data[2];

    // Process STATUS BYTE
    _log_status_byte(*param_ptr_status_byte);

    // Check status byte: error bit
    if ((*param_ptr_status_byte & MLX90393_STATUS_ERROR_BITMASK) != 0)
    {
        return HAL_ERROR;
    }

    return HAL_OK;
}

/*********************************************************************************
 * _write_register()
 *
 *  Write directly in the volatile RAM
 *********************************************************************************/
static HAL_StatusTypeDef _write_register(const mlx90393_config_t *param_ptr_config,
                                         mlx90393_reg_t param_reg, uint16_t param_data,
                                         mlx90393_status_byte_t *param_ptr_status_byte)
{
    HAL_StatusTypeDef ret = HAL_OK;
    uint8_t tx_data[4] = {
        MLX90393_CMD_WRITE_REGISTER,
        (uint8_t)(param_data >> 8),   // MSB
        (uint8_t)(param_data & 0xFF), // LSB
        param_reg << 2};
    uint8_t rx_data[1] = {0};

    // Send write register command
    ret = HAL_I2C_Master_Transmit(param_ptr_config->i2c_handle,
                                  param_ptr_config->i2c_slave_addr << 1,
                                  tx_data, 4, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK)
    {
        return ret;
    }

    // Receive status byte
    ret = HAL_I2C_Master_Receive(param_ptr_config->i2c_handle,
                                 param_ptr_config->i2c_slave_addr << 1,
                                 rx_data, 1, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK)
    {
        return ret;
    }

    *param_ptr_status_byte = rx_data[0];

    // Process STATUS BYTE
    _log_status_byte(*param_ptr_status_byte);

    // Check status byte: error bit
    if ((*param_ptr_status_byte & MLX90393_STATUS_ERROR_BITMASK) != 0)
    {
        return HAL_ERROR;
    }

    return HAL_OK;
}

/*********************************************************************************
 * mlx90393_init()
 *
 * Initialize the MLX90393 sensor
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_init(mlx90393_config_t* param_ptr_config) {
    HAL_StatusTypeDef ret = HAL_OK;

    /*
     * I2C
     *
     * @important In STM32, I2C is typically configured during peripheral initialization
     *            so we assume the I2C handle is already properly configured
     */
    if (param_ptr_config->manage_i2c_driver) {
        // In STM32, I2C is usually initialized separately, so we just validate the handle
        if (param_ptr_config->i2c_handle == NULL || param_ptr_config->i2c_handle->State == HAL_I2C_STATE_RESET) {
            LOG_Error("I2C handle not properly initialized");
            return HAL_ERROR;
        }
    }

    /*
     * INT DRDY Data Ready pin: GPIO
     *   @rule GPIO_PIN_NONE means not used to detect that a measurement is ready to be read.
     */
    if (param_ptr_config->int_gpio_pin != 0) {
        GPIO_InitTypeDef GPIO_InitStruct = {0};
        // Configure GPIO pin
        GPIO_InitStruct.Pin = param_ptr_config->int_gpio_pin;
        GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
        GPIO_InitStruct.Pull = GPIO_PULLDOWN; // @important
        HAL_GPIO_Init(param_ptr_config->int_gpio_port, &GPIO_InitStruct);
        
        LOG_Info("I2C slave addr 0x%X", param_ptr_config->i2c_slave_addr);
    } else {
        HAL_Delay(10);
    }

    /*
     * Commands
     */

    // @doc The exit command is used to force the IC into idle mode.
    ret = mlx90393_cmd_exit(param_ptr_config);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_cmd_exit() failed");
        return ret;
    }

    // Reset the device (this implicitly verifies that the I2C slave device is working properly)
    ret = mlx90393_cmd_reset(param_ptr_config);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_cmd_reset() failed");
        return ret;
    }

    /*
     * Getting MLX NVRAM param values (some are read-only)
     */
    // SENS_TC_LT
    uint8_t sens_tc_lt;
    ret = mlx90393_get_sens_tc_lt(param_ptr_config, &sens_tc_lt);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_sens_tc_lt() failed");
        return ret;
    }
    param_ptr_config->mlx_sens_tc_lt = sens_tc_lt;

    // SENS_TC_HT
    uint8_t sens_tc_ht;
    ret = mlx90393_get_sens_tc_ht(param_ptr_config, &sens_tc_ht);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_sens_tc_ht() failed");
        return ret;
    }
    param_ptr_config->mlx_sens_tc_ht = sens_tc_ht;

    // TREF (16bit, storing this readonly register value for converting raw metrics to functional metrics
    uint16_t tref;
    ret = mlx90393_get_tref(param_ptr_config, &tref);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_tref() failed");
        return ret;
    }
    param_ptr_config->mlx_tref = tref;

    /*
     * Setting good parameter values
     */
    // COMM_MODE
    ret = mlx90393_set_comm_mode(param_ptr_config, MLX90393_COMM_MODE_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_comm_mode() failed");
        return ret;
    }

    // TCMP_EN
    ret = mlx90393_set_tcmp_en(param_ptr_config, MLX90393_TCMP_EN_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_tcmp_en() failed");
        return ret;
    }

    // HALLCONF
    ret = mlx90393_set_hallconf(param_ptr_config, MLX90393_HALLCONF_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_hallconf() failed");
        return ret;
    }

    // GAIN_SEL
    ret = mlx90393_set_gain_sel(param_ptr_config, MLX90393_GAIN_SEL_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_gain_sel() failed");
        return ret;
    }

    // OSR
    ret = mlx90393_set_osr(param_ptr_config, MLX90393_OSR_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_osr() failed");
        return ret;
    }

    // DIG_FILT
    ret = mlx90393_set_dig_filt(param_ptr_config, MLX90393_DIG_FILT_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_dig_filt() failed");
        return ret;
    }

    // RES_XYZ
    ret = mlx90393_set_res_xyz(param_ptr_config, MLX90393_RES_XYZ_DEFAULT, MLX90393_RES_XYZ_DEFAULT, MLX90393_RES_XYZ_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_res_xyz() failed");
        return ret;
    }

    // OFFSET_X OFFSET_Y OFFSET_Z
    ret = mlx90393_set_offset_xyz(param_ptr_config, MLX90393_OFFSET_X_DEFAULT, MLX90393_OFFSET_Y_DEFAULT, MLX90393_OFFSET_Z_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_set_offset_xyz() failed");
        return ret;
    }
    uint8_t n = 0, m = 0;
    if(param_ptr_config->mlx_metrics_selector.x_axis)
        n++;
    if(param_ptr_config->mlx_metrics_selector.y_axis)
        n++;
    if(param_ptr_config->mlx_metrics_selector.z_axis)
        n++;
    if(param_ptr_config->mlx_metrics_selector.temperature)
        m++;
    param_ptr_config->conv_time = (uint16_t)(1 + 
                                    n * 0.063f *(1 << param_ptr_config->mlx_osr) * 
                                    (2 + (1<<param_ptr_config->mlx_dig_filt)) +
                                    0.18f * 4); // in ms
    uint16_t compensation_time = 0;
    param_ptr_config->conv_time = param_ptr_config->conv_time - compensation_time > 2 ?
                        param_ptr_config->conv_time - compensation_time : 3;         
    return HAL_OK;
}

/*********************************************************************************
 * mlx90393_deinit()
 *
 * Deinitialize the MLX90393 sensor
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_deinit(const mlx90393_config_t* param_ptr_config) {
    HAL_StatusTypeDef ret = HAL_OK;

    /*
     * I2C Driver
     * In STM32, I2C deinitialization is typically handled elsewhere
     */
    if (param_ptr_config->manage_i2c_driver) {
        // In STM32, we typically don't deinitialize I2C here as it might be used by other devices
        // If needed, this would be handled at the application level
    }

    return ret;
}

/*********************************************************************************
 * mlx90393_log_device_parameters()
 *
 * Log all device parameters
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_log_device_parameters(const mlx90393_config_t* param_ptr_config) {
    HAL_StatusTypeDef ret = HAL_OK;

    LOG_Info("MLX90393 LOG DEVICE PARAMETERS (*READ AGAIN FROM REGISTERS*):");

    // COMM_MODE
    mlx90393_comm_mode_t comm_mode;
    ret = mlx90393_get_comm_mode(param_ptr_config, &comm_mode);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_comm_mode() failed");
        return ret;
    }
    LOG_Info("  COMM_MODE: 0x%X (%u)", comm_mode, comm_mode);

    // TCMP_EN
    mlx90393_tcmp_en_t tcmp_en;
    ret = mlx90393_get_tcmp_en(param_ptr_config, &tcmp_en);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_tcmp_en() failed");
        return ret;
    }
    LOG_Info("  TCMP_EN: 0x%X (%u)", tcmp_en, tcmp_en);

    // HALLCONF
    mlx90393_hallconf_t hallconf;
    ret = mlx90393_get_hallconf(param_ptr_config, &hallconf);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_hallconf() failed");
        return ret;
    }
    LOG_Info("  HALLCONF: 0x%X (%u)", hallconf, hallconf);

    // GAIN_SEL
    mlx90393_gain_sel_t gain_sel;
    ret = mlx90393_get_gain_sel(param_ptr_config, &gain_sel);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_gain_sel() failed");
        return ret;
    }
    LOG_Info("  GAIN_SEL: 0x%X (%u)", gain_sel, gain_sel);

    // Z_SERIES Only get()
    mlx90393_z_series_t z_series;
    ret = mlx90393_get_z_series(param_ptr_config, &z_series);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_z_series() failed");
        return ret;
    }
    LOG_Info("  Z_SERIES: 0x%X (%u)", z_series, z_series);

    // BIST Only get()
    mlx90393_bist_t bist;
    ret = mlx90393_get_bist(param_ptr_config, &bist);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_bist() failed");
        return ret;
    }
    LOG_Info("  BIST: 0x%X (%u)", bist, bist);

    // EXT_TRIG Only get()
    mlx90393_ext_trig_t ext_trig;
    ret = mlx90393_get_ext_trig(param_ptr_config, &ext_trig);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_ext_trig() failed");
        return ret;
    }
    LOG_Info("  EXT_TRIG: 0x%X (%u)", ext_trig, ext_trig);

    // TRIG_INT_SEL Only get()
    mlx90393_trig_int_sel_t trig_int_sel;
    ret = mlx90393_get_trig_int_sel(param_ptr_config, &trig_int_sel);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_trig_int_sel() failed");
        return ret;
    }
    LOG_Info("  TRIG_INT_SEL: 0x%X (%u)", trig_int_sel, trig_int_sel);

    // OSR
    mlx90393_osr_t osr;
    ret = mlx90393_get_osr(param_ptr_config, &osr);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_osr() failed");
        return ret;
    }
    LOG_Info("  OSR: 0x%X (%u)", osr, osr);

    // DIG_FILT
    mlx90393_dig_filt_t dig_filt;
    ret = mlx90393_get_dig_filt(param_ptr_config, &dig_filt);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_dig_filt() failed");
        return ret;
    }
    LOG_Info("  DIG_FILT: 0x%X (%u)", dig_filt, dig_filt);

    // RES_XYZ
    mlx90393_res_xyz_t res_x, res_y, res_z;
    ret = mlx90393_get_res_xyz(param_ptr_config, &res_x, &res_y, &res_z);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_res_xyz() failed");
        return ret;
    }
    LOG_Info("  RES_XYZ: X= 0x%X (%u) | Y= 0x%X (%u) | Z= 0x%X (%u)", res_x, res_x, res_y, res_y, res_z, res_z);

    // SENS_TC_LT Only get()
    uint8_t sens_tc_lt;
    ret = mlx90393_get_sens_tc_lt(param_ptr_config, &sens_tc_lt);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_sens_tc_lt() failed");
        return ret;
    }
    LOG_Info("  SENS_TC_LT: 0x%X (%u)", sens_tc_lt, sens_tc_lt);

    // SENS_TC_HT Only get()
    uint8_t sens_tc_ht;
    ret = mlx90393_get_sens_tc_ht(param_ptr_config, &sens_tc_ht);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_sens_tc_ht() failed");
        return ret;
    }
    LOG_Info("  SENS_TC_HT: 0x%X (%u)", sens_tc_ht, sens_tc_ht);

    // OFFSET_X OFFSET_Y OFFSET_Z Only get()
    uint16_t offset_x, offset_y, offset_z;
    ret = mlx90393_get_offset_xyz(param_ptr_config, &offset_x, &offset_y, &offset_z);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_offset_xyz() failed");
        return ret;
    }
    LOG_Info("  OFFSET_*: X= 0x%X (%u) | Y= 0x%X (%u) | Z= 0x%X (%u)", offset_x, offset_x, offset_y, offset_y, offset_z, offset_z);

    // TREF
    uint16_t tref;
    ret = mlx90393_get_tref(param_ptr_config, &tref);
    if (ret != HAL_OK) {
        LOG_Error("mlx90393_get_tref() failed");
        return ret;
    }
    LOG_Info("  TREF: 0x%X (%u)", tref, tref);

    return HAL_OK;
}

/*********************************************************************************
 * cmd_reset()
 *
 * This command is used to reset the IC. On reset, the idle mode will be entered again.
 * The status byte will reflect that the reset has been successful.
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_cmd_reset(const mlx90393_config_t* param_ptr_config) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status_byte = 0;

    ret = _send_cmd(param_ptr_config, MLX90393_CMD_RESET, &status_byte);
    if (ret != HAL_OK) {
        LOG_Error("cmd_reset() failed");
        return ret;
    }

    // SPECIFIC: Check status byte: reset bit must be set
    if ((status_byte & MLX90393_STATUS_RESET_BITMASK) == 0) {
        LOG_Error("The STATUS BYTE's RESET BIT is not set");
        return HAL_ERROR;
    }

    // @important Delay. Data sheet "TPOR Power-on-reset completion time = 1.5 millisec"
    HAL_Delay(5);

    // INFORM
    LOG_Debug("The device has been reset :)");
    HAL_Delay(10);

    return HAL_OK;
}

/*********************************************************************************
 * get_comm_mode()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_comm_mode(const mlx90393_config_t* param_ptr_config, mlx90393_comm_mode_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_COMM_MODE_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_COMM_MODE_BITMASK) >> MLX90393_COMM_MODE_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_tcmp_en()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_tcmp_en(const mlx90393_config_t* param_ptr_config, mlx90393_tcmp_en_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_TCMP_EN_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_TCMP_EN_BITMASK) >> MLX90393_TCMP_EN_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_hallconf()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_hallconf(const mlx90393_config_t* param_ptr_config, mlx90393_hallconf_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_HALLCONF_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_HALLCONF_BITMASK) >> MLX90393_HALLCONF_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_gain_sel()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_gain_sel(const mlx90393_config_t* param_ptr_config, mlx90393_gain_sel_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_GAIN_SEL_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_GAIN_SEL_BITMASK) >> MLX90393_GAIN_SEL_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_z_series()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_z_series(const mlx90393_config_t* param_ptr_config, mlx90393_z_series_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_Z_SERIES_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_Z_SERIES_BITMASK) >> MLX90393_Z_SERIES_BITSHIFT;

    return HAL_OK;
}


/*********************************************************************************
 * cmd_exit()
 *
 * The exit command is used to force the IC into idle mode.
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_cmd_exit(const mlx90393_config_t* param_ptr_config) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status_byte = 0;

    ret = _send_cmd(param_ptr_config, MLX90393_CMD_EXIT_MODE, &status_byte);
    if (ret != HAL_OK) {
        LOG_Error("_send_cmd() failed");
        return ret;
    }

    return HAL_OK;
}

/*********************************************************************************
 * get_bist()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_bist(const mlx90393_config_t* param_ptr_config, mlx90393_bist_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_BIST_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_BIST_BITMASK) >> MLX90393_BIST_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_ext_trig()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_ext_trig(const mlx90393_config_t* param_ptr_config, mlx90393_ext_trig_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_EXT_TRIG_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_EXT_TRIG_BITMASK) >> MLX90393_EXT_TRIG_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_trig_int_sel()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_trig_int_sel(const mlx90393_config_t* param_ptr_config, mlx90393_trig_int_sel_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_TRIG_INT_SEL_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_TRIG_INT_SEL_BITMASK) >> MLX90393_TRIG_INT_SEL_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_osr()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_osr(const mlx90393_config_t* param_ptr_config, mlx90393_osr_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_OSR_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_OSR_BITMASK) >> MLX90393_OSR_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_dig_filt()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_dig_filt(const mlx90393_config_t* param_ptr_config, mlx90393_dig_filt_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_DIG_FILT_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_DIG_FILT_BITMASK) >> MLX90393_DIG_FILT_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_res_xyz()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_res_xyz(const mlx90393_config_t* param_ptr_config, mlx90393_res_xyz_t* param_x, mlx90393_res_xyz_t* param_y,
                                   mlx90393_res_xyz_t* param_z) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_RES_XYZ_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    uint8_t res_zyx = (reg_data & MLX90393_RES_XYZ_BITMASK) >> MLX90393_RES_XYZ_BITSHIFT;

    // Extract X Y Z 2-bit values from parameter value
    *param_x = (res_zyx >> 0) & 0x3;
    *param_y = (res_zyx >> 2) & 0x3;
    *param_z = (res_zyx >> 4) & 0x3;

    return HAL_OK;
}

/*********************************************************************************
 * get_sens_tc_lt()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_sens_tc_lt(const mlx90393_config_t* param_ptr_config, uint8_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_SENS_TC_LT_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_SENS_TC_LT_BITMASK) >> MLX90393_SENS_TC_LT_BITSHIFT;

    // Adjust with extra drift due to the QFN packaging (data sheet)
    *param_ptr_data -= 6;

    return HAL_OK;
}

/*********************************************************************************
 * get_sens_tc_ht()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_sens_tc_ht(const mlx90393_config_t* param_ptr_config, uint8_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_SENS_TC_HT_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_SENS_TC_HT_BITMASK) >> MLX90393_SENS_TC_HT_BITSHIFT;

    // Adjust with extra drift due to the QFN packaging (data sheet)
    *param_ptr_data -= 15;

    return HAL_OK;
}

/*********************************************************************************
 * get_offset_xyz()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_offset_xyz(const mlx90393_config_t* param_ptr_config, uint16_t* param_x, uint16_t* param_y, uint16_t* param_z) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // X
    ret = _read_register(param_ptr_config, MLX90393_OFFSET_X_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }
    *param_x = (reg_data & MLX90393_OFFSET_X_BITMASK) >> MLX90393_OFFSET_X_BITSHIFT;

    // Y
    ret = _read_register(param_ptr_config, MLX90393_OFFSET_Y_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }
    *param_y = (reg_data & MLX90393_OFFSET_Y_BITMASK) >> MLX90393_OFFSET_Y_BITSHIFT;

    // Z
    ret = _read_register(param_ptr_config, MLX90393_OFFSET_Z_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }
    *param_z = (reg_data & MLX90393_OFFSET_Z_BITMASK) >> MLX90393_OFFSET_Z_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * get_tref()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_get_tref(const mlx90393_config_t* param_ptr_config, uint16_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    ret = _read_register(param_ptr_config, MLX90393_TREF_THRESHOLD_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Extract parameter value from register data
    *param_ptr_data = (reg_data & MLX90393_TREF_THRESHOLD_BITMASK) >> MLX90393_TREF_THRESHOLD_BITSHIFT;

    return HAL_OK;
}

/*********************************************************************************
 * set_comm_mode()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_comm_mode(mlx90393_config_t* param_ptr_config, mlx90393_comm_mode_t param_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // READ
    ret = _read_register(param_ptr_config, MLX90393_COMM_MODE_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_COMM_MODE_BITMASK) | ((param_data << MLX90393_COMM_MODE_BITSHIFT) & MLX90393_COMM_MODE_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_COMM_MODE_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_comm_mode = param_data;

    return HAL_OK;
}

/*********************************************************************************
 * set_tcmp_en()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_tcmp_en(mlx90393_config_t* param_ptr_config, mlx90393_tcmp_en_t param_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // Input params
    if (param_data == MLX90393_TCMP_EN_ENABLED) {
        LOG_Error("The setting TCMP_EN=1 (temperature compensation enabled) is not supported");
        return HAL_ERROR;
    }

    // READ
    ret = _read_register(param_ptr_config, MLX90393_TCMP_EN_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_TCMP_EN_BITMASK) | ((param_data << MLX90393_TCMP_EN_BITSHIFT) & MLX90393_TCMP_EN_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_TCMP_EN_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_tcmp_en = param_data;

    return HAL_OK;
}

/*********************************************************************************
 * set_hallconf()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_hallconf(mlx90393_config_t* param_ptr_config, mlx90393_hallconf_t param_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // Input params
    if (param_data == MLX90393_HALLCONF_0) {
        LOG_Error("The setting MJD_MLX90393_HALLCONF=0x0 is not supported");
        return HAL_ERROR;
    }

    // READ
    ret = _read_register(param_ptr_config, MLX90393_HALLCONF_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_HALLCONF_BITMASK) | ((param_data << MLX90393_HALLCONF_BITSHIFT) & MLX90393_HALLCONF_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_HALLCONF_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_hallconf = param_data;

    return HAL_OK;
}

/*********************************************************************************
 * set_gain_sel()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_gain_sel(mlx90393_config_t* param_ptr_config, mlx90393_gain_sel_t param_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // READ
    ret = _read_register(param_ptr_config, MLX90393_GAIN_SEL_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_GAIN_SEL_BITMASK) | ((param_data << MLX90393_GAIN_SEL_BITSHIFT) & MLX90393_GAIN_SEL_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_GAIN_SEL_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_gain_sel = param_data;

    return HAL_OK;
}

/*********************************************************************************
 * set_osr()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_osr(mlx90393_config_t* param_ptr_config, mlx90393_osr_t param_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // READ
    ret = _read_register(param_ptr_config, MLX90393_OSR_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_OSR_BITMASK) | ((param_data << MLX90393_OSR_BITSHIFT) & MLX90393_OSR_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_OSR_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_osr = param_data;

    return HAL_OK;
}

/*********************************************************************************
 * set_dig_filt()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_dig_filt(mlx90393_config_t* param_ptr_config, mlx90393_dig_filt_t param_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    // READ
    ret = _read_register(param_ptr_config, MLX90393_DIG_FILT_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_DIG_FILT_BITMASK) | ((param_data << MLX90393_DIG_FILT_BITSHIFT) & MLX90393_DIG_FILT_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_DIG_FILT_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_dig_filt = param_data;

    return HAL_OK;
}

/*********************************************************************************
 * set_res_xyz()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_res_xyz(mlx90393_config_t* param_ptr_config, mlx90393_res_xyz_t param_res_x, mlx90393_res_xyz_t param_res_y,
                                   mlx90393_res_xyz_t param_res_z) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data, zyx_data; // word

    // READ
    ret = _read_register(param_ptr_config, MLX90393_RES_XYZ_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Combine the Z Y X 2-bit values from the xyz input params
    zyx_data = ((param_res_z & 0x3) << 4) | ((param_res_y & 0x3) << 2) | ((param_res_x & 0x3) << 0);

    // Inject new data
    reg_data = (reg_data & ~MLX90393_RES_XYZ_BITMASK) | ((zyx_data << MLX90393_RES_XYZ_BITSHIFT) & MLX90393_RES_XYZ_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_RES_XYZ_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    // SAVE IN CONFIG STRUCT
    param_ptr_config->mlx_res_x = param_res_x;
    param_ptr_config->mlx_res_y = param_res_y;
    param_ptr_config->mlx_res_z = param_res_z;

    return HAL_OK;
}

/*********************************************************************************
 * set_offset_xyz()
 *
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_set_offset_xyz(mlx90393_config_t* param_ptr_config, uint16_t param_offset_x, uint16_t param_offset_y,
                                      uint16_t param_offset_z) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;
    uint16_t reg_data; // word

    /*
     * OFFSET_X
     */
    // READ
    ret = _read_register(param_ptr_config, MLX90393_OFFSET_X_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_OFFSET_X_BITMASK) | ((param_offset_x << MLX90393_OFFSET_X_BITSHIFT) & MLX90393_OFFSET_X_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_OFFSET_X_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    /*
     * OFFSET_Y
     */
    // READ
    ret = _read_register(param_ptr_config, MLX90393_OFFSET_Y_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_OFFSET_Y_BITMASK) | ((param_offset_y << MLX90393_OFFSET_Y_BITSHIFT) & MLX90393_OFFSET_Y_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_OFFSET_Y_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    /*
     * OFFSET_Z
     */
    // READ
    ret = _read_register(param_ptr_config, MLX90393_OFFSET_Z_REG, &status, &reg_data);
    if (ret != HAL_OK) {
        LOG_Error("_read_register() failed");
        return ret;
    }

    // Inject new data
    reg_data = (reg_data & ~MLX90393_OFFSET_Z_BITMASK) | ((param_offset_z << MLX90393_OFFSET_Z_BITSHIFT) & MLX90393_OFFSET_Z_BITMASK);

    // WRITE
    ret = _write_register(param_ptr_config, MLX90393_OFFSET_Z_REG, reg_data, &status);
    if (ret != HAL_OK) {
        LOG_Error("_write_register() failed");
        return ret;
    }

    /*
     * SAVE IN CONFIG STRUCT
     */
    param_ptr_config->mlx_offset_x = param_offset_x;
    param_ptr_config->mlx_offset_y = param_offset_y;
    param_ptr_config->mlx_offset_z = param_offset_z;

    return HAL_OK;
}

/*********************************************************************************
 * cmd_start_measurement()
 *
 * The single measurement command is used to instruct the MLX90393 to perform an acquisition cycle.
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_cmd_start_measurement(const mlx90393_config_t* param_ptr_config) {
    HAL_StatusTypeDef ret = HAL_OK;
    mlx90393_status_byte_t status;

    // Convert config metrics flags to the command's LSNibble syntax
    uint8_t zyxt_nibble = 0;
    if (param_ptr_config->mlx_metrics_selector.x_axis == true) {
        zyxt_nibble |= MLX90393_METRIC_X_AXIS_BITMASK;
    }
    if (param_ptr_config->mlx_metrics_selector.y_axis == true) {
        zyxt_nibble |= MLX90393_METRIC_Y_AXIS_BITMASK;
    }
    if (param_ptr_config->mlx_metrics_selector.z_axis == true) {
        zyxt_nibble |= MLX90393_METRIC_Z_AXIS_BITMASK;
    }
    if (param_ptr_config->mlx_metrics_selector.temperature == true) {
        zyxt_nibble |= MLX90393_METRIC_TEMPERATURE_BITMASK;
    }

    ret = _send_cmd(param_ptr_config, MLX90393_CMD_START_SINGLE_MEASUREMENT_MODE | zyxt_nibble, &status);
    if (ret != HAL_OK) {
        LOG_Error("_send_cmd() failed");
        return ret;
    }

    // SPECIFIC: Check status byte: the special SM bit
    if ((status & MLX90393_STATUS_SINGLE_MEASUREMENT_MODE_BITMASK) == 0) {
        LOG_Error("The STATUS BYTE's SINGLE_MEASUREMENT_MODE_BITMASK is not set");
        return HAL_ERROR;
    }

    /*
     * THE DEVICE NEEDS TIME TO COLLECT AND SET READY ITS MEASUREMENTS
     * Wait for measurement ready. Either using the INT DRDY Data Ready pin or a long fixed waiting time
     * @rule int_gpio_num = -1 means the pin is not used to detect that a measurement is ready to be read.
     *
     * When not using the INT DRDY Data Ready pin: 'worst case', when all parameters are max and ZYXT equals 0b1111, the conversion time is 198.5ms.
     *
     */
    if (param_ptr_config->int_gpio_pin == 0) {
        // @important A LONG SPECIFIC FIXED DELAY.
        // HAL_Delay(200); // 200ms delay for worst-case scenario
    } else {
        // WAIT for the INT DRDY Data Ready pin value go to 1 XOR timeout
        // bool has_timed_out = false;
        // const uint32_t MLX_TIMEOUT_MS = 2000; // 2 seconds timeout
        // uint32_t start_time = HAL_GetTick();
        // while (HAL_GPIO_ReadPin(param_ptr_config->int_gpio_port, param_ptr_config->int_gpio_pin) != GPIO_PIN_SET) {
        //     if (HAL_GetTick() - start_time > MLX_TIMEOUT_MS) {
        //         has_timed_out = true;
        //         break;
        //     }
        //     HAL_Delay(10); // Wait in increments of 10 milliseconds
        // }

        // if (has_timed_out == false) {
        //     LOG_Info("OK. The INT DRDY Data Ready pin value did go to 1");
        // } else {
        //     LOG_Error("The INT DRDY Data Ready pin value did not go to 1 after the time out");
        //     return HAL_ERROR;
        // }
    }

    return HAL_OK;
}

/*********************************************************************************
 * cmd_read_measurement()
 *
 * Read measurement data from the MLX90393 sensor
 *********************************************************************************/
HAL_StatusTypeDef mlx90393_cmd_read_measurement(const mlx90393_config_t* param_ptr_config, mlx90393_data_t* param_ptr_data) {
    HAL_StatusTypeDef ret = HAL_OK;
    uint8_t metrics_nbr_of_bytes_to_read = 0;
    uint8_t command;
    mlx90393_status_byte_t status;
    static uint8_t buf_rx[9] = {0};  // Max 9 bytes that any MLX command will return via I2C

    metrics_nbr_of_bytes_to_read = 0;
    // Input params: convert config metrics flags to the command's LSNibble syntax
    uint8_t zyxt_nibble = 0;
    if (param_ptr_config->mlx_metrics_selector.temperature == true) {
        zyxt_nibble |= MLX90393_METRIC_TEMPERATURE_BITMASK;
        ++metrics_nbr_of_bytes_to_read;
    }
    if (param_ptr_config->mlx_metrics_selector.x_axis == true) {
        zyxt_nibble |= MLX90393_METRIC_X_AXIS_BITMASK;
        ++metrics_nbr_of_bytes_to_read;
    }
    if (param_ptr_config->mlx_metrics_selector.y_axis == true) {
        zyxt_nibble |= MLX90393_METRIC_Y_AXIS_BITMASK;
        ++metrics_nbr_of_bytes_to_read;
    }
    if (param_ptr_config->mlx_metrics_selector.z_axis == true) {
        zyxt_nibble |= MLX90393_METRIC_Z_AXIS_BITMASK;
        ++metrics_nbr_of_bytes_to_read;
    }

    // Input param .mlx_metrics_selector: compute the number of BYTES to read next via I2C
    // Each metric is 1 word (2 bytes via I2C)
    // At least one metric must be selected for read
    metrics_nbr_of_bytes_to_read *= 2; // #words => #bytes
    if (metrics_nbr_of_bytes_to_read < 2) {
        LOG_Error("At least one metric must be selected for read");
        return HAL_ERROR;
    }

    // Send request & Receive response: STATUS BYTE + METRICS DATA (variable number of bytes)
    command = MLX90393_CMD_READ_MEASUREMENT | zyxt_nibble;

    // Prepare transmission
    uint8_t tx_data[1] = {command};
    uint8_t rx_data[9] = {0}; // Status byte + up to 8 data bytes

    // Send command
    ret = HAL_I2C_Master_Transmit(param_ptr_config->i2c_handle, param_ptr_config->i2c_slave_addr << 1, tx_data, 1, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("I2C transmit failed");
        return ret;
    }

    // Receive response: status byte + metrics data
    ret = HAL_I2C_Master_Receive(param_ptr_config->i2c_handle, param_ptr_config->i2c_slave_addr << 1, rx_data, 1 + metrics_nbr_of_bytes_to_read, MLX90393_I2C_TIMEOUT_DEFAULT);
    if (ret != HAL_OK) {
        LOG_Error("I2C receive failed");
        return ret;
    }

    // Extract status byte
    status = rx_data[0];

    // Process STATUS BYTE: check error bit
    // _log_status_byte(status);
    if ((status & MLX90393_STATUS_ERROR_BITMASK) != 0) {
        LOG_Error("The STATUS BYTE's ERROR BIT is set");
        return HAL_ERROR;
    }

    // Process METRICS DATA (raw)
    // The device data is output in the following order: T (MSB), T (LSB), X (MSB), X (LSB), Y (MSB), Y (LSB), Z (MSB), Z (LSB)
    mlx90393_data_raw_t data_raw = {0};
    uint8_t idx = 1; // Start after status byte

    if (param_ptr_config->mlx_metrics_selector.temperature == true) {
        data_raw.t = (((uint16_t)rx_data[idx] << 8) | (uint16_t)rx_data[idx + 1]);
        idx += 2;
    } else {
        data_raw.t = 0;
    }
    if (param_ptr_config->mlx_metrics_selector.x_axis == true) {
        data_raw.x = (((uint16_t)rx_data[idx] << 8) | (uint16_t)rx_data[idx + 1]);
        idx += 2;
    } else {
        data_raw.x = 0;
    }
    if (param_ptr_config->mlx_metrics_selector.y_axis == true) {
        data_raw.y = (((uint16_t)rx_data[idx] << 8) | (uint16_t)rx_data[idx + 1]);
        idx += 2;
    } else {
        data_raw.y = 0;
    }
    if (param_ptr_config->mlx_metrics_selector.z_axis == true) {
        data_raw.z = (((uint16_t)rx_data[idx] << 8) | (uint16_t)rx_data[idx + 1]);
    } else {
        data_raw.z = 0;
    }


    // Save raw data in the final data structure
    param_ptr_data->t_raw = data_raw.t;
    param_ptr_data->x_raw = data_raw.x;
    param_ptr_data->y_raw = data_raw.y;
    param_ptr_data->z_raw = data_raw.z;

    // Transform metrics data (raw -> functional depending on system settings)
    // T
    param_ptr_data->t = 35 + (data_raw.t - param_ptr_config->mlx_tref) / 45.2f;

    // XYZ
    switch (param_ptr_config->mlx_res_x) {
    case MLX90393_RES_XYZ_0:
    case MLX90393_RES_XYZ_1:
        param_ptr_data->x = ((int16_t)(data_raw.x) * _sensitivity_xy[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_x] * (1 << (param_ptr_config->mlx_res_x)));
        break;
    case MLX90393_RES_XYZ_2:
        param_ptr_data->x = (data_raw.x - 0x8000) * _sensitivity_xy[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_x] * (1 << param_ptr_config->mlx_res_x);
        break;
    case MLX90393_RES_XYZ_3:
        param_ptr_data->x = (data_raw.x - 0x4000) * _sensitivity_xy[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_x] * (1 << param_ptr_config->mlx_res_x);
        break;
    default:
        break;
    }

    switch (param_ptr_config->mlx_res_y) {
    case MLX90393_RES_XYZ_0:
    case MLX90393_RES_XYZ_1:
        param_ptr_data->y = ((int16_t)(data_raw.y) * _sensitivity_xy[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_y] * (1 << (param_ptr_config->mlx_res_y)));
        break;
    case MLX90393_RES_XYZ_2:
        param_ptr_data->y = (data_raw.y - 0x8000) * _sensitivity_xy[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_y] * (1 << param_ptr_config->mlx_res_y);
        break;
    case MLX90393_RES_XYZ_3:
        param_ptr_data->y = (data_raw.y - 0x4000) * _sensitivity_z[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_y] * (1 << param_ptr_config->mlx_res_y);
        break;
    default:
        break;
    }

    switch (param_ptr_config->mlx_res_z) {
    case MLX90393_RES_XYZ_0:
    case MLX90393_RES_XYZ_1:
        param_ptr_data->z = ((int16_t)(data_raw.z) * _sensitivity_z[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_z] * (1 << (param_ptr_config->mlx_res_z)));
        break;
    case MLX90393_RES_XYZ_2:
        param_ptr_data->z = (data_raw.z - 0x8000) * _sensitivity_z[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_z] * (1 << param_ptr_config->mlx_res_z);
        break;
    case MLX90393_RES_XYZ_3:
        param_ptr_data->z = (data_raw.z - 0x4000) * _sensitivity_z[param_ptr_config->mlx_gain_sel][param_ptr_config->mlx_res_z] * (1 << param_ptr_config->mlx_res_z);
        break;
    default:
        break;
    }

    return HAL_OK;
}