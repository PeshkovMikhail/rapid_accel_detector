import av
import cv2
import threading
import queue
import time

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import BufferedInputFile
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.exceptions import TelegramBadRequest
from aiogram.utils.markdown import hbold
import io
import numpy as np
import os, copy
import gc

from preprocess import *
from config import *
from speed_tracker import SpeedTracker
from action_detector import ActionDetector

transforms = [
    UniformSampleFrames(POSEC3D_INPUT_FRAMES_COUNT),
    PoseDecode(),
    PoseCompact(hw_ratio=1., allow_imgpad=True),
    Resize(scale=(-1, 64)),
    CenterCrop(crop_size=64),
    GeneratePoseTarget(sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    FormatShape(input_format='NCTHW_Heatmap'),
    PackActionInputs()
]

task_queue = queue.Queue()

def get_label(session, seq):
    temp = copy.deepcopy(seq)
    for transform in transforms:
        temp = transform.transform(temp)
    input_feed = {'input_tensor': temp["inputs"].cpu().data.numpy()}
    outputs = session.run(['cls_score'], input_feed=input_feed)
    return np.argmax(torch.nn.functional.softmax(torch.tensor(outputs[0][0])).numpy())

async def send_result(id, output_memory_file):
    logging.warning("entered send_result function")
    await bot.send_video(id, BufferedInputFile(output_memory_file.read(), "output.mp4"))

def models_thread() :
    while True:
        if not task_queue.empty():
            task = task_queue.get()

            frame_id = 0
            input_file = task['input_file']
            input_file.seek(0)
            container = av.open(input_file)
            in_stream = container.streams.video[0]

            height = in_stream.height
            width = in_stream.width
            fps = np.floor(container.streams.video[0].average_rate)
            framebuffer = np.zeros((POSEC3D_INPUT_FRAMES_COUNT, height, width, 3), np.uint8)
            
            speed_tracker = SpeedTracker(height, width, 1 / fps)
            action_detector = ActionDetector(height, width, "posec3d.onnx")

            output_memory_file = io.BytesIO()
            output_f = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
            stream = output_f.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
            speed_per_frame = []

            for frame in container.decode(video = 0):
                img_np = np.array(frame.to_image(), dtype=np.uint8)
            
                framebuffer[frame_id % POSEC3D_INPUT_FRAMES_COUNT] = img_np.copy()

                current_speed = action_detector.update_poses(img_np, frame_id)
                speed_per_frame.append(speed_tracker.speed(current_speed, frame_id))

                if (frame_id + 1) % POSEC3D_INPUT_FRAMES_COUNT == 0:
                    classes = action_detector.classify()
                    for f in range(POSEC3D_INPUT_FRAMES_COUNT):
                        img = framebuffer[f]
                        for k, (l, b) in classes.items():
                            color = (255, 0, 0)
                            if l == 1:
                                color = (0, 255, 0)
                            pt1 = np.array((
                                int(b[f][0] - b[f][2]/2),
                                int(b[f][1] - b[f][3]/2)
                            ))
                            pt2 = np.array((
                                int(b[f][0] + b[f][2]/2),
                                int(b[f][1] + b[f][3]/2)
                            ))
                            if k in speed_per_frame[f].keys():
                                img = cv2.putText(img, f"{speed_per_frame[f][k]*3.6:.2f} km/h", pt1 - np.array([0, 20]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA, False)
                                img = cv2.rectangle(img, pt1, pt2, color, 3)
                        res_img = av.VideoFrame.from_ndarray(img, format='rgb24') # Convert image from NumPy Array to frame.
                        packet = stream.encode(res_img)  # Encode video frame
                        output_f.mux(packet)
                    speed_per_frame = []
                frame_id += 1
            packet = stream.encode(None)
            output_f.mux(packet)
            output_f.close()
            output_memory_file.seek(0) 
            send_fut = asyncio.run_coroutine_threadsafe(asyncio.run_coroutine_threadsafe(send_result(task['id'], output_memory_file), loop=task['loop']))
            send_fut.result()
            del output_memory_file
            del input_file
            gc.collect()
            task_queue.task_done()
        time.sleep(2)



# Bot token can be obtained via https://t.me/BotFather
TOKEN = os.getenv('TELEGRAM_TOKEN')

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()
bot = Bot(TOKEN, parse_mode=ParseMode.HTML)

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    kb = [
        [
            types.KeyboardButton(text="YOLOv8"),
            types.KeyboardButton(text="ViTPose")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True
    )
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {hbold(message.from_user.full_name)}!")

@dp.message(F.text.lower() == "YOLOv8")
async def use_yolo():
    POSE_DETECTOR = "yolo"

@dp.message(F.text.lower() == "ViTPose")
async def use_vit():
    POSE_DETECTOR = "vitpose"

@dp.message()
async def echo_handler(message: types.Message) -> None:
    # Send a copy of the received message
    if not message.video:
        await message.answer("Video required")
        return

    file_id = message.video.file_id
    try:
        file = await bot.get_file(file_id)
    except TelegramBadRequest as e:
        await message.answer(f"{e}")
        return
    file_path = file.file_path
    print(file_path)
    

    task = {}
    task['input_file'] = await bot.download_file(file_path)
    task['id'] = message.chat.id
    task['loop'] = asyncio.get_event_loop()
    await message.answer("Processing video")
    task_queue.put(task)
    


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    threading.Thread(target=models_thread, daemon=True).start()
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())