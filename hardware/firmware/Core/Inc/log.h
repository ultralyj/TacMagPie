/**
 * @file log.h
 * @author Luo Yijie (2331857@tongji.edu.cn)
 * @brief Header file of log module.
 * @version 0.1
 * @date 2024-08-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef __LOG_H_
#define __LOG_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdarg.h>
#include <stdio.h>
#include "stm32g0xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */
typedef enum
{
	LOG_LEVEL_DEBUG = 0,
	LOG_LEVEL_INFO,
	LOG_LEVEL_WARN,
	LOG_LEVEL_ERROR,
}E_LOGLEVEL;
/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/

/* USER CODE BEGIN EFP */
char* LOG_GetLevel(const int level);
void LOG_Print(const int level, const char* fun, const int line, const char* fmt, ...);
#define LOG_Printf(level,fmt,...) LOG_Print(level,__FUNCTION__,__LINE__,fmt, ##__VA_ARGS__)
#define LOG_Error(fmt,...) LOG_Print(LOG_LEVEL_ERROR,__FUNCTION__,__LINE__,fmt, ##__VA_ARGS__)
#define LOG_Warn(fmt,...) LOG_Print(LOG_LEVEL_WARN,__FUNCTION__,__LINE__,fmt, ##__VA_ARGS__)
#define LOG_Info(fmt,...) LOG_Print(LOG_LEVEL_INFO,__FUNCTION__,__LINE__,fmt, ##__VA_ARGS__)
#define LOG_Debug(fmt,...) LOG_Print(LOG_LEVEL_DEBUG,__FUNCTION__,__LINE__,fmt, ##__VA_ARGS__)

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
/* USER CODE BEGIN Private defines */
#define OPEN_LOG 1
#define LOG_LEVEL LOG_LEVEL_INFO
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif


