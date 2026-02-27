/**
 * @file reprintf.c
 * @author Luo-Yijie (1951578@tongji.edu.cn)
 * @brief redirect printf to serial port
 * @version 0.1
 * @date 2022-12-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "stm32g0xx_hal.h"
#include <stdio.h>

#define STDIO_PORT huart2

extern UART_HandleTypeDef STDIO_PORT;


/**
 * @brief redirect printf to serial port
 *
 */
int fputc(int ch, FILE *f)
{
    HAL_UART_Transmit(&STDIO_PORT, (uint8_t *)&ch, 1, 0xffff);
    while (__HAL_UART_GET_FLAG(&STDIO_PORT, UART_FLAG_TC) == RESET)
    {
    }
    return ch;
}

