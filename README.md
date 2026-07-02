# MuMu ADB VC

本项目是一个本地模拟器 UI 自动化框架，用于学习、UI 自动化测试和个人辅助操作。它通过 ADB 连接 MuMu 模拟器，使用截图和 OpenCV 图像识别判断当前状态，再通过标准 ADB 输入执行点击或滑动。

## 安全边界

- 只使用 `adb shell input tap`、`adb shell input swipe`、`adb shell input keyevent` 等标准输入。
- 状态判断只基于截图图像识别。
- 不读取 App 内存，不抓包，不调用私有接口，不注入，不 Hook，不修改游戏文件。
- 不实现反检测、绕过风控、绕过反作弊、批量账号、协议模拟或模拟登录态。
- 客户端需要手动点击开始，提供暂停、停止、定时启动和日志，不做系统后台静默运行。

## 目录约定

```text
MuMu ADB VC/
  app.py                       # CLI 入口
  client.py                    # 桌面客户端
  build_exe.py                 # 固定构建 release/MuMuADBVC/MuMuADBVC.exe
  config.yaml                  # ADB、MuMu、识别坐标配置
  tasks/                       # 任务状态机 YAML
  routes/                      # 路线导航 YAML
  assets/templates/            # 小模板图
  assets/routes/               # 路线关键帧整屏图
  debug/                       # 截图、模板源图、匹配调试图
  logs/                        # 文本日志和 jsonl 事件日志
  tests/                       # 单元测试
```

打包后固定使用：

```text
release/MuMuADBVC/MuMuADBVC.exe
```

`release/MuMuADBVC/` 会一起复制 `config.yaml`、`tasks/`、`routes/`、`assets/`、`debug/`、`logs/`。

## 安装和构建

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-build.txt
python -m pytest
python build_exe.py
```

## MuMu 连接配置

当前配置参考 MAA 的 MuMu 连接方式：

```yaml
adb_path: 'C:\Program Files\NetEase\MuMu Player 12\nx_device\12.0\shell\adb.exe'
device_serial: "127.0.0.1:16384"
emulator_path: 'C:\Program Files\NetEase\MuMu Player 12\nx_main\MuMuNxMain.exe'
auto_launch_emulator: true
game_package: com.tencent.tmgp.sgame
close_game_after_run: true
recognition_width: 1280
recognition_height: 720
```

运行任务时会先检查 ADB 截图是否可用；不可用时会尝试启动 MuMu 并等待连接。`devices` 只列设备，不主动启动模拟器。

## 常用命令

```powershell
python app.py devices
python app.py capture
python app.py capture --recognition --output debug/template_sources/current.png
python app.py match --template assets/templates/wzry/one_key_farm.png
python app.py run --task tasks/wzry_farm.yaml
python app.py client
```

优先使用客户端：双击 `release/MuMuADBVC/MuMuADBVC.exe`，在界面里填 ADB 地址、ADB 路径、MuMu 路径，点击截图、开始、暂停、停止。

## 模板采集

客户端里已经有“模板采集”区域：

1. 让模拟器停在目标画面。
2. 在下拉框选择当前阶段。
3. 点击“采集当前屏幕”。
4. 点击“打开素材目录”，从 `debug/template_sources/` 中裁剪小模板。
5. 点击“打开模板目录”，把裁剪后的 PNG 放到 `assets/templates/wzry/`。

模板只裁剪稳定 UI 元素，例如开始游戏按钮、右上角关闭按钮、农场入口文字、仓库/种植/百科图标、一键务农按钮。不要裁剪账号昵称等会变化的内容。

## 王者荣耀农场流程

当前任务文件是 `tasks/wzry_farm.yaml`，流程为：

1. 回到安卓桌面。
2. 识别并点击王者荣耀图标。
3. 等待“开始游戏”界面，识别成功后点击。
4. 循环关闭大厅弹窗。
5. 识别“来农场干农活”入口并点击。
6. 等待农场加载完成，使用仓库、社交、种植、百科、返回等多个稳定 UI 模板确认场景。
7. 农场内再检测一次右上角关闭按钮，处理进入农场后才弹出的活动页。
8. 使用 `route_navigate` 固定左上断步移动靠近雕像互动点。
9. 识别“一键务农”按钮并点击。
10. 如果出现“恭喜您获得”收获页，就点击底部空白处继续。
11. 继续向左上方断步移动到作物上，读取左侧作物卡片里的成熟时间，例如 `13:04成熟`。
12. 如果勾选【定时启动】，根据作物类型和成熟时间安排下一轮最优浇水/收获；任务完成或失败后会用标准 ADB 关闭游戏后台。

任务支持从中途接管。重新点“开始”时，会先识别当前画面是弹窗、开始游戏、大厅入口、农场、还是一键务农按钮附近，再从对应步骤继续。

## 客户端 UI

客户端左侧是任务选项，右侧小齿轮会把对应说明显示到中间设置区：

- 【开始唤醒】：保存后，下次打开客户端会自动执行已勾选任务。
- 【一键务农】：当前已实现的完整农场流程。
- 【领取奖励】和【自动偷菜】：预留入口，暂未接入执行逻辑。

## 定时启动和成熟时间

客户端勾选【定时启动】后，定时器只在客户端窗口打开期间工作，不注册 Windows 后台服务，也不会静默常驻。每次任务完成后：

- 如果在【下次启动】里填写 `HH:MM`，客户端会自动武装这一次手动定时；点击【应用】会立即校验格式。该时间只覆盖下一次启动，触发后自动清空，并会保存到 `client_state.yaml`，客户端重开后仍可继续等待。
- 【作物类型】支持 `1h`、`8h`、`16h`、`32h`。自动调度按作物总时长 `T` 计算：首次浇水减少 `T/12`，水分维持 `T/3`，进入 `5/12T` 阈值后按剩余时间 `* 0.8` 安排最后一次启动，目标是接近 `11/15T` 的最短收获时间。
- 如果成功读取到作物成熟时间，会把屏幕上的 `HH:MM成熟` 或 `明天HH:MM成熟` 作为当前实际剩余时间参与调度；`明天` 前缀会保留，避免把次日凌晨误判成今天。读不到时使用上一次任务记录的作物剩余时间继续推算。
- 任务完成后，右侧日志会输出“一键务农最优解”，列出后续每一次定时启动时间、预计剩余成熟时间，以及最快收获时间。
- 每次完整任务结束、任务失败或卡死后，会执行 `adb shell am force-stop com.tencent.tmgp.sgame` 关闭王者荣耀后台，下一轮再从模拟器桌面/游戏入口重新启动。

## 农场移动逻辑

当前 `routes/wzry_farm.yaml` 使用 `mode: fixed_step`。进入农场后不再做关键帧路线判断，而是执行写死的闭环：

- 每轮先截图检查弹窗。
- 再检查“一键务农”按钮，识别到就结束移动，交给下一节点点击。
- 没有识别到“一键务农”时，只向左上方拖动摇杆一次。
- 每次断步移动后等待至少 `8000ms`，给画面稳定和按钮出现留窗口期。
- 如果连续 `300000ms` 仍然没有到达交互点，就识别并点击右下角农场重置按钮，让人物回到原位，然后继续向左上断步移动。
- 点击“一键务农”后，成熟时间读取也使用同样的左上断步思路：每次移动后等待 `8000ms`，直到左侧作物卡片出现。读取时会先用金色铲子按钮模板 `assets/templates/wzry/maturity_shovel_button.png` 锚定卡片，再读取上方的成熟时间，支持 `HH:MM成熟` 和 `明天HH:MM成熟`，最长等待约 10 分钟。

核心配置：

```yaml
mode: fixed_step
joystick:
  center: [184, 505]
  distance: 72
  duration_ms: 280
fixed_step:
  direction: up_left
  step_wait_ms: 8000
  reset_after_ms: 300000
reset:
  templates:
    - wzry/farm_reset_button.png
  threshold: 0.70
  roi: [1080, 560, 190, 150]
```

重置按钮模板在 `assets/templates/wzry/farm_reset_button.png`。如果后续界面皮肤或背景导致识别不稳，可以重新从右下角圆形重置按钮裁一张更干净的小模板。

## 日志

客户端左侧显示流程进度，右侧输出运行日志。关键字段包括：

- 客户端右侧日志只显示用户需要看的进度，例如“当前步骤：点击开始游戏”“完成：点击一键务农”“作物成熟时间：13:04”“下次定时启动：12:54”。
- 右侧日志会用绿色显示成功步骤，用红色显示失败步骤；任务失败、卡死或异常时会保存 `debug/failure_snapshots/` 快照，并在日志里提供“查看卡死截图”链接。
- 详细模板分数、坐标、debug 画框仍写入 `logs/` 和 `debug/`，用于排查问题。
- 每次任务结束后会自动清理运行截图缓存，保留 `debug/template_sources/` 手动采集素材和 `debug/failure_snapshots/` 失败快照。客户端【清理截图缓存】按钮可手动执行同样的清理。

日志文件在 `logs/`，调试截图在 `debug/`。
