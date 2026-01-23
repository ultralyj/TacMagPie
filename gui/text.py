# simple_sensor_server_fixed.py
import asyncio
import json
import random
from datetime import datetime
import websockets
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleSensorServer:
    def __init__(self, num_sensors=5):
        self.bx = 0
        self.by = 0
        self.bz = 0.5
        self.num_sensors = num_sensors
        self.connected_clients = set()
        self.running = True
    
    def generate_sensor_data(self):
        """生成模拟传感器数据"""
        data = {}
        
        for i in range(1, self.num_sensors + 1):
            # 生成随机磁场数据，模拟实际传感器的变化
            # Bx, By, Bz 范围在 -2.0 到 2.0 特斯拉之间
            if i == 1:
                # 第一个传感器使用完全随机值
                self.bx += random.uniform(-0.05, 0.05)
                self.by += random.uniform(-0.05, 0.05)
                self.bz += random.uniform(-0.05, 0.05)
            else:
                # 让相邻传感器的数据有一定关联性
                self.bx = data[f"B{i-1}x"] + random.uniform(-0.2, 0.2)
                self.by = data[f"B{i-1}y"] + random.uniform(-0.2, 0.2)
                self.bz = data[f"B{i-1}z"] + random.uniform(-0.2, 0.2)
            
            data[f"B{i}x"] = round(self.bx, 3)
            data[f"B{i}y"] = round(self.by, 3)
            data[f"B{i}z"] = abs(round(self.bz, 3))
        
        # 添加时间戳
        data["timestamp"] = datetime.now().isoformat()
        data["sensor_count"] = self.num_sensors
        
        return data
    
    async def handler(self, websocket):
        """处理客户端连接 - 参数已简化为只有一个websocket"""
        client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}" if websocket.remote_address else "unknown"
        logger.info(f"新客户端连接: {client_address}")
        self.connected_clients.add(websocket)
        
        try:
            # 发送连接确认
            await websocket.send(json.dumps({
                "status": "connected",
                "message": "开始接收传感器数据"
            }))
            
            # 等待客户端消息，保持连接
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    if msg.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开连接: {client_address}")
        except Exception as e:
            logger.error(f"处理错误: {e}")
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
    
    async def broadcast_data(self):
        """定期发送数据到所有连接的客户端"""
        while self.running:
            if self.connected_clients:
                data = self.generate_sensor_data()
                message = json.dumps(data)
                
                # 记录断开连接的客户端
                disconnected = []
                
                # 发送给所有连接的客户端
                for client in self.connected_clients:
                    try:
                        await client.send(message)
                    except:
                        disconnected.append(client)
                
                # 移除断开连接的客户端
                for client in disconnected:
                    if client in self.connected_clients:
                        self.connected_clients.remove(client)
                        
                logger.debug(f"向 {len(self.connected_clients)} 个客户端发送了数据")
            
            # 等待0.5秒
            await asyncio.sleep(0.05)
    
    async def start(self, host="localhost", port=8765):
        """启动服务器"""
        # 启动WebSocket服务器
        server = await websockets.serve(self.handler, host, port)
        logger.info(f"服务器启动在 ws://{host}:{port}")
        
        # 启动数据广播任务
        broadcast_task = asyncio.create_task(self.broadcast_data())
        
        try:
            # 保持服务器运行
            await server.wait_closed()
        finally:
            self.running = False
            await broadcast_task

# 运行服务器
if __name__ == "__main__":
    server = SimpleSensorServer(num_sensors=5)
    
    print("=" * 50)
    print("磁传感器模拟服务器")
    print(f"WebSocket地址: ws://localhost:8765")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n服务器已停止")