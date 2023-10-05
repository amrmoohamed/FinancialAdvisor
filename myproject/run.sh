#!/bin/bash
python manage.py runserver &
sleep 5
open -a "Google Chrome" http://127.0.0.1:8000/chatbot/