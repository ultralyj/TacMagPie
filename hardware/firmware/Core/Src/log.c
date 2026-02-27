/**
 * @file log.c
 * @author Luo Yijie (2331857@tongji.edu.cn)
 * @brief This file provides code for the log.
 * @version 0.1
 * @date 2024-08-29
 *
 * @copyright Copyright (c) 2024
 *
 */
 
#ifdef __cplusplus
extern "C"
{
#endif

/* Includes ------------------------------------------------------------------*/
#include "log.h"

char* LOG_GetLevel(const int level)
{
	if (level == LOG_LEVEL_DEBUG) {
		return "DEBUG";
	}
	else if (level == LOG_LEVEL_INFO) {
		return "INFO";
	}
	else if (level == LOG_LEVEL_WARN) {
		return "WARN";
	}
	else if (level == LOG_LEVEL_ERROR) {
		return "ERROR";
	}
	return "UNLNOW";
}

void LOG_Print(const int level,const char* fun, const int line ,const char* fmt, ...)
{
#ifdef OPEN_LOG
	va_list arg;
	va_start(arg, fmt);
	static char buf[500] = { 0 };
	vsnprintf(buf, sizeof(buf), fmt, arg);
	va_end(arg);
	if (level >= LOG_LEVEL)	
		printf("[%06d] %-5s | [%s] %s \r\n", HAL_GetTick(), LOG_GetLevel(level), fun, buf);
#endif
}


#ifdef __cplusplus
}
#endif
	








