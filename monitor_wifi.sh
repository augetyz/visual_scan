#!/bin/bash

# WiFi 名称和密码
SSID="biubiu"
PASSWORD="123456789"

# 无限循环，后台持续监测
while true; do
    # 检查是否连接到目标 WiFi
    current_ssid=$(nmcli -t -f ACTIVE,SSID dev wifi | grep '^yes' | cut -d':' -f2)
    
    if [ "$current_ssid" != "$SSID" ]; then
        echo "未连接到 '$SSID'，尝试连接..."
        nmcli device wifi connect "$SSID" password "$PASSWORD"
        
        if [ $? -eq 0 ]; then
            echo "已成功连接到 '$SSID'。"
        else
            echo "连接 '$SSID' 失败，请检查网络设置。"
        fi
    else
        echo "已连接到 '$SSID'。"
    fi

    # 每隔 30 秒监测一次
    sleep 30
done
