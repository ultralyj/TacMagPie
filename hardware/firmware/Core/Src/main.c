/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "i2c.h"
#include "iwdg.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "mlx90393.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_NVIC_Init(void);
/* USER CODE BEGIN PFP */
void I2C_ScanDevices(void);
void MATRIX_Update_Measurement(uint8_t sub_baseline);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#define SENSOR_NUM 5
#define MATRIX_I2C_ADDR_BASE 0x0c
#define MATRIX_EMPTY_ROUND 5
#define MATRIX_BASELINE_ROUND 10

static mlx90393_config_t mlx90393_handle[SENSOR_NUM];
static mlx90393_metrics_selector_t mlx90393_selector = {
    .x_axis = true,
    .y_axis = true,
    .z_axis = true,
    .temperature = false};
mlx90393_data_t g_baseline[SENSOR_NUM] = {0}; // To store the baseline values for x, y, z
mlx90393_data_t g_data[SENSOR_NUM];
char LOG_Buffer[256];

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C2_Init();
  MX_IWDG_Init();
  MX_TIM2_Init();
  MX_TIM14_Init();
  MX_USART2_UART_Init();
  MX_I2C1_Init();
  MX_TIM16_Init();

  /* Initialize interrupts */
  MX_NVIC_Init();
  /* USER CODE BEGIN 2 */
  printf("hello\r\n");
  I2C_ScanDevices();
  /* Configurate the basic parameters of mlx90393 */
  for (uint8_t i = 0; i < SENSOR_NUM; i++)
  {
    mlx90393_handle[i].manage_i2c_driver = false;
    mlx90393_handle[i].i2c_handle = i == 0 ? (&hi2c2) : (&hi2c1);
    mlx90393_handle[i].i2c_slave_addr = i == 0 ? MATRIX_I2C_ADDR_BASE : MATRIX_I2C_ADDR_BASE + i - 1;
    mlx90393_handle[i].int_gpio_port = 0;
    mlx90393_handle[i].int_gpio_pin = 0;
    mlx90393_handle[i].mlx_metrics_selector = mlx90393_selector;

    /* Initialize the sensor */
    if (mlx90393_init(&mlx90393_handle[i]) != HAL_OK)
    {
      LOG_Error("mlx90393 (%d) init failed i2c=%s addr=0x%02d", i, mlx90393_handle[i].i2c_handle == (&hi2c1) ? "i2c1" : "i2c2", mlx90393_handle[i].i2c_slave_addr);
    }
  }

  for (uint16_t round = 0; round < MATRIX_EMPTY_ROUND; round++)
  {
    MATRIX_Update_Measurement(false);
  }

  for (uint8_t round = 0; round < MATRIX_BASELINE_ROUND; round++)
  {
    MATRIX_Update_Measurement(false);
    for (uint8_t i = 0; i < SENSOR_NUM; i++)
    {
      g_baseline[i].x += g_data[i].x;
      g_baseline[i].y += g_data[i].y;
      g_baseline[i].z += g_data[i].z;
    }
  }
  /* Calculate the average baseline */
  for (uint8_t i = 0; i < SENSOR_NUM; i++)
  {
    g_baseline[i].x /= MATRIX_BASELINE_ROUND;
    g_baseline[i].y /= MATRIX_BASELINE_ROUND;
    g_baseline[i].z /= MATRIX_BASELINE_ROUND;
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    MATRIX_Update_Measurement(true);
    sprintf(LOG_Buffer, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\r\n",
            g_data[0].x,g_data[0].y,g_data[0].z,
            g_data[1].x,g_data[1].y,g_data[1].z,
            g_data[2].x,g_data[2].y,g_data[2].z,
            g_data[3].x,g_data[3].y,g_data[3].z,
            g_data[4].x,g_data[4].y,g_data[4].z);
    HAL_UART_Transmit_DMA(&huart2, (uint8_t *)LOG_Buffer, strlen(LOG_Buffer));
    HAL_IWDG_Refresh(&hiwdg);
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
   */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI | RCC_OSCILLATORTYPE_LSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.LSIState = RCC_LSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 8;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
 * @brief NVIC Configuration.
 * @retval None
 */
static void MX_NVIC_Init(void)
{
  /* I2C1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(I2C1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(I2C1_IRQn);
}

/* USER CODE BEGIN 4 */
void I2C_ScanDevices(void)
{
  uint8_t found_devices = 0;
  LOG_Info("Starting I2C scan...");

  for (uint8_t addr = 0x01; addr < 0x20; addr++)
  {
    HAL_StatusTypeDef status;
    status = HAL_I2C_IsDeviceReady(&hi2c1, addr << 1, 10, 10);
    if (status == HAL_OK)
    {
      LOG_Info("Device found at address: 0x%02X", addr);
      found_devices++;
    }
    else if (status == HAL_TIMEOUT)
    {
      LOG_Info("Timeout at address: 0x%02X", addr);
    }
    HAL_Delay(1);
  }
  LOG_Info("Scan complete. Found %d device(s).", found_devices);
}

void MATRIX_Update_Measurement(uint8_t sub_baseline)
{
  for (uint8_t i = 0; i < SENSOR_NUM; i++)
  {
    if (mlx90393_cmd_start_measurement(&mlx90393_handle[i]) != HAL_OK)
    {
      LOG_Error("start measurement failed i2c=%s addr=0x%02d", mlx90393_handle[i].i2c_handle == (&hi2c1) ? "i2c1" : "i2c2", mlx90393_handle[i].i2c_slave_addr);
    }
  }
  HAL_Delay(mlx90393_handle[0].conv_time);
  for (uint8_t i = 0; i < SENSOR_NUM; i++)
  {
    if (mlx90393_cmd_read_measurement(&mlx90393_handle[i], &(g_data[i])) != HAL_OK)
    {
      LOG_Error("read measurement failed i2c=%s addr=0x%02d", mlx90393_handle[i].i2c_handle == (&hi2c1) ? "i2c1" : "i2c2", mlx90393_handle[i].i2c_slave_addr);
    }
  }
  if (sub_baseline)
  {
    for (uint8_t i = 0; i < SENSOR_NUM; i++)
    {
      g_data[i].x -= g_baseline[i].x;
      g_data[i].y -= g_baseline[i].y;
      g_data[i].z -= g_baseline[i].z;
    }
  }
}
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
