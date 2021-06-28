import os, io, logging
import torch, random
import torchvision.transforms as transforms
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
from aiogram.types.message import ContentType
from aiogram.types import ParseMode, InputMediaPhoto, \
        ChatActions, ReplyKeyboardRemove

from config import TOKEN
from messages import MESSAGES
import keyboards as kb
from style_transfer_NN import StyleTransferModel
from MSGNet import load_MSGNet, tensor_save_bgrimage

class States(StatesGroup):
    content_simpleNN = State()
    style_simpleNN = State()
    content_MSG = State()
    style_MSG = State()

bot = Bot(token = TOKEN)
memory_storage = MemoryStorage()
dp = Dispatcher(bot, storage = memory_storage)

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

model = StyleTransferModel()
MSG_model = load_MSGNet()


@dp.message_handler(commands = ['start'], state = "*")
async def process_start_command(message: types.Message):
    await message.reply(MESSAGES['start'])


@dp.message_handler(commands = ['help'], state = "*")
async def process_help_command(message: types.Message):
    await message.reply(MESSAGES['help'], reply_markup = kb.actions_kb,
                            parse_mode = ParseMode.MARKDOWN)


@dp.message_handler(commands=['cancel'], state = "*")
async def cancel_state(message: types.Message, state: FSMContext):
    # finishing the current state, so now user can start all over again
    await state.finish()

    await message.answer(random.choice(MESSAGES['cancel']),
                            reply_markup = kb.actions_kb)


@dp.message_handler(commands = ['simpleST'], state = "*")
async def process_begin_command(message: types.Message):
    # setting the state in which user is expected to send content photo
    await States.content_simpleNN.set()

    await message.answer(random.choice(MESSAGES['photo_request']),
                            reply_markup = kb.cancel_kb)


@dp.message_handler(commands = ['MSGNetST'], state = "*")
async def process_begin_command(message: types.Message):
    # setting the state in which user is expected to send content photo
    await States.content_MSG.set()

    await message.answer(random.choice(MESSAGES['photo_request']),
                            reply_markup = kb.cancel_kb)


@dp.message_handler(state = States.content_simpleNN, content_types = ContentType.ANY)
async def get_content_image(message: types.Message, state: FSMContext):
    # this handler is triggered when user is expected to send a content photo,
    # so if this is not the case, we send 'wrong input' message
    if not message.photo:
        await message.answer(random.choice(MESSAGES['wrong_input']))
        return

    # adding photo id in state inforamtion
    await state.update_data(content_img_id = message.photo[-1].file_id)

    # setting the state in which user is expected to send style photo
    await States.style_simpleNN.set()


@dp.message_handler(state = States.content_MSG, content_types = ContentType.ANY)
async def get_content_image(message: types.Message, state: FSMContext):
    # this handler is triggered when user is expected to send a content photo,
    # so if this is not the case, we send 'wrong input' message
    if not message.photo:
        await message.answer(random.choice(MESSAGES['wrong_input']))
        return

    # adding photo id in state inforamtion
    await state.update_data(content_img_id = message.photo[-1].file_id)

    # setting the state in which user is expected to send style photo
    await States.style_MSG.set()


@dp.message_handler(state = States.style_simpleNN, content_types = ContentType.ANY)
async def get_style_image(message: types.Message, state: FSMContext):
    # this handler is triggered when user is expected to send a style photo,
    # so if this is not the case, we send 'wrong input' message
    if not message.photo:
        await message.answer(random.choice(MESSAGES['wrong_input']))
        return

    # adding photo id and user id in state inforamtion
    await state.update_data(style_img_id = message.photo[-1].file_id)
    await state.update_data(user_id = message.from_user.id)

    await message.answer(random.choice(MESSAGES['runST']),
                            reply_markup=ReplyKeyboardRemove())

    # extracting state information and finishing the state
    inputs = await state.get_data()
    await state.finish()

    await start_style_transfer(inputs, 'simple')


@dp.message_handler(state = States.style_MSG, content_types = ContentType.ANY)
async def get_style_image(message: types.Message, state: FSMContext):
    # this handler is triggered when user is expected to send a style photo,
    # so if this is not the case, we send 'wrong input' message
    if not message.photo:
        await message.answer(random.choice(MESSAGES['wrong_input']))
        return

    # adding photo id and user id in state inforamtion
    await state.update_data(style_img_id = message.photo[-1].file_id)
    await state.update_data(user_id = message.from_user.id)

    await message.answer(random.choice(MESSAGES['runST']),
                            reply_markup=ReplyKeyboardRemove())

    # extracting state information and finishing the state
    inputs = await state.get_data()
    await state.finish()

    await start_style_transfer(inputs, 'MSG')


async def start_style_transfer(inputs, mode):
    # getting content and style images from state information (inputs)
    content_img_info = await bot.get_file(inputs['content_img_id'])
    style_img_info = await bot.get_file(inputs['style_img_id'])

    content_img = await bot.download_file(content_img_info.file_path)
    style_img = await bot.download_file(style_img_info.file_path)

    # starting the style transfer algorithm depending on the mode
    if mode == 'simple':
        output = await model.run_style_transfer(content_img, style_img)
    elif mode == 'MSG':
        output = await MSG_model.run_style_transfer(content_img, style_img)

    await send_output(inputs, output, mode)


async def send_output(inputs, output, mode):
    # converting image from tensor
    if mode == 'simple':
        PIL_transform = transforms.ToPILImage()
        img_tensor = output.cpu().squeeze(0)
        img = PIL_transform(img_tensor)
    else:
        img = tensor_save_bgrimage(output.data[0], False)

    # saving image
    image = io.BytesIO()
    img.save(image, format = 'JPEG')
    img = image.getvalue()

    await types.ChatActions.upload_photo()
    await bot.send_photo(inputs['user_id'], img, random.choice(MESSAGES['result']))

    # sending a suggestion to do something else
    await bot.send_message(inputs['user_id'], MESSAGES['action_request'],
                            reply_markup = kb.inline_kb)


@dp.callback_query_handler(text = 'yes')
async def process_callback_yes_btn(callback_query: types.CallbackQuery):
    # this handler is triggered if user accepted action suggestion
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id,
                            MESSAGES['action_choice'],
                            reply_markup = kb.actions_kb)


@dp.callback_query_handler(text = 'no')
async def process_callback_no_btn(callback_query: types.CallbackQuery):
    # this handler is triggered if user didn't accept action suggestion
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, MESSAGES['action_refuse'])


@dp.message_handler(content_types = ContentType.ANY)
async def unknown_message(msg: types.Message):
    await msg.reply(random.choice(MESSAGES['unknown_message']),
                        parse_mode = ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)
