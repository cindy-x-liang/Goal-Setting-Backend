# Goal Setting Backend

The goal setting app is powered by a Flask backend. 
SQLAlchemy database is used to store user's goals. The backend uses sentiment analysis to get a numerical rating on the user's journal entries. 
Users may also evaluate their progress on their goalswhich is ran by OpenAI API supported agent. 

User's can also chat with a virtual therapist bot which uses whisper and text-to-speech libraries in python. 

Demo and more detailed description on frontend github repo:https://github.com/cindy-x-liang/Goal-Setting-Frontend

# How to run
1. make sure your OpenAI API key is set at os.environ["OPENAI_API_KEY"] =
2. create a virtual environment
3. install pyttsx3, openai, flask, ntlk, pandas, SpeechRecognition
4. run python server.py
