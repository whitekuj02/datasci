from os import environ
from Bard import Chatbot

token = environ.get("V669vpftMeIoyeEo/AhojFu5_uBK2s1_c8")

chatbot = Chatbot(token)

chatbot.ask("Hello, how are you?")