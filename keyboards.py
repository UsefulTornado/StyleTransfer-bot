from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, \
        InlineKeyboardMarkup, InlineKeyboardButton

actions_kb = ReplyKeyboardMarkup(
    resize_keyboard = True, one_time_keyboard = True
).add(KeyboardButton('/simpleST'),
        KeyboardButton('/MSGNetST'))

cancel_kb = ReplyKeyboardMarkup(
    resize_keyboard = True
).add(KeyboardButton('/cancel'))

inline_kb = InlineKeyboardMarkup().add(
    InlineKeyboardButton('Да!', callback_data='yes'),
    InlineKeyboardButton('Не-а', callback_data='no'))
