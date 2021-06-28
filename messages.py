from aiogram.utils.markdown import text, bold, italic, code
from aiogram.utils.emoji import emojize

start_message = emojize('Привет! :v:\n' \
                'Я бот, что был рождён под гибельной звездой, ' \
                'c желаньями безбрежными, как вечность. :dizzy:\n' \
                'Я могу переносить стиль с одной картинки на другую. :raised_hands:\n' \
                'Используйте /help, чтобы узнать список доступных команд!')

help_message = text(bold('Я могу ответить на следующие команды:\n'),
                bold('/simpleST'),
                ' - перенос стиля с помощью простого алгоритма.'
                ' Алгоритм выполнится в течение 3 минут.\n',
                bold('/MSGNetST'),
                ' - перенос стиля с помощью Multi-style Generative Network.'
                ' Данная сеть заточена на обработку в художественном стиле.'
                ' В среднем результат будет готов уже через 5-7 секунд.\n',
                bold('/cancel'),
                ' - отмена текущих действий и возврат к исходному состоянию.\n',
                sep = '')

cancel_messages = [
                emojize('Можно начинать заново :recycle:'),
                'Заново так заново',
                'Всё успешно было забыто',
                'Сделаю вид, что ничего не было',
                'Всё по-новой...',
                emojize('Ваше желание исполнено! :sparkles:')
]

photo_request_messages = [
                'Отправьте мне фото, на которое нужно перенести стиль, ' \
                'и фото со стилем',

                emojize('Заклинаю отослать мне фото, ' \
                'что послужит основой для дел моих тёмных, ' \
                'и фото, чей образ затмит душу первой картинки :smiling_imp:'),

                emojize('Во имя света давайте же скрестим фотографии с контентом ' \
                'и со стилем! :zap: Отправьте их в соответствующем порядке!'),

                'Нужно отправить две картинки, ' \
                'на первую из которых хотелось бы перенести стиль второй',

                'Отправьте основную фотографию и фотографию со стилем'
]

wrong_input_messages = [
                'Всё же нужно фото',
                'Нужна фотография',
                emojize('Но я.. я ведь.. просил.. фотографию.. :unamused:'),
                'Мне бы всё-таки фотографию...',
                'Вы так шутите?)',
                emojize('Если это фото, то я живой человек :grimacing:'),
                'Не принимается',
                'Не-не-не, я ведь попросил фотографии',
                'Хотелось бы фото',
                'Не издевайтесь...',
                'Просил ведь фото... Обидно..'
]

run_ST_messages = [
                'Начинаю перенос стиля...',
                'Запускаю алгоритм..',
                emojize('Поехали! :sunglasses:'),
                'Скоро всё будет готово!'
]

result_messages = [
                emojize('Сделано! :fire:'),
                'Вот и всё!',
                'Готово!',
                'Вот что у меня получилось',
                'Как-то так..'
]

request_to_action_message = 'Попробуем что-нибудь ещё?'

action_choice_message = emojize('Отлично! Что будем делать в этот раз? :smirk:')

action_refuse_message = 'Ну ладно...'

unknown_input_messages = [
                text(emojize('Я не знаю, что с этим делать :astonished:'),
                    italic('\nЯ просто напомню,'), 'что есть',
                    code('команда'), '/help'),
                'Это что-то на эльфийском? Используйте команду /help',
                emojize(':hushed: К такому я не был готов!'
                        ' Попробуйте использовать /help'),
                'Я в ступоре.. Может команда /help поможет?',
                'Не понимаю.. Используйте /help'
]

MESSAGES = {
    'start': start_message,
    'help': help_message,
    'cancel': cancel_messages,
    'photo_request': photo_request_messages,
    'wrong_input': wrong_input_messages,
    'runST': run_ST_messages,
    'result': result_messages,
    'action_request': request_to_action_message,
    'action_choice': action_choice_message,
    'action_refuse': action_refuse_message,
    'unknown_message': unknown_input_messages
}
