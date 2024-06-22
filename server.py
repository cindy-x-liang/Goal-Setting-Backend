import json
import threading
import time
from db import db
from flask import Flask, request, jsonify
from db import Goals
from flask_cors import CORS
import nltk
import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)
db_filename = "goals.db"
nltk.download('all')
dontwannachangedb = '1'

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///%s" % db_filename
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = True

db.init_app(app)
with app.app_context():
    db.create_all()

def success_response(data, code=200):
    return json.dumps(data)


def failure_response(message, code=404):
    return json.dumps({"error": message}), code

import getpass
import os



from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
from langchain_core.messages import HumanMessage


log = ''


@app.route("/progress/<int:goal_id>/")
def get_goal_progress(goal_id):
    """
    Endpoint for getting all tasks
    """
    global log
    course = Goals.query.filter_by(id=goal_id).first() 
    if course is None:
        return failure_response("Goal not found!") 
    prompt='evaluate my goal progress for this goal' + course.title+'where i made this progress' + course.progress + 'this is the description of the goal' + course.description + 'i want to finish by' + course.endDate + 'and this is how im feeling' + log
    response = model.invoke([HumanMessage(content=prompt)])
    print(response.content)
    returnVal = {}
    returnVal['message'] = response.content
    return jsonify(returnVal), 201

    


@app.route("/")
@app.route("/goals")
def get_goals():
    """
    Endpoint for getting all tasks
    """
    global dontwannachangedb

    goals = Goals.query.all()
    courses = [course.to_dict() for course in Goals.query.all()] 
    toReturn  = [dontwannachangedb] + courses
    return success_response(toReturn) 
    #return success_response({"goals": courses}) 
    # return jsonify([goal.to_dict() for goal in goals])

@app.route("/goals", methods=["POST"])
def create_goals():
    """
    Endpoint for creating a new task
    """
    body = json.loads(request.data)
    new_course = Goals(
        title = body.get("title"),
        progress = "0",
        description = body.get("description"),
        startDate = body.get("startDate"),
        endDate = body.get("endDate")
    )
    # if not body.get("code"):
    #     return failure_response("Sender not found",400)
    
    # if not body.get("name"):
    #     return failure_response("Sender not found",400)

    db.session.add(new_course) 
    db.session.commit() 
    return jsonify(new_course.to_dict()), 201

@app.route("/goals/<int:goal_id>/")
def get_goal(goal_id):
    """
    Endpoint for getting a task by id
    """
    course = Goals.query.filter_by(id=goal_id).first() 
    if course is None:
        return failure_response("Goal not found!") 
    return jsonify(course.to_dict()), 201

def preprocess_text(text):

    # Tokenize the text

    tokens = word_tokenize(text.lower())
    # Remove stop words

    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string

    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# apply the function df

#df['text'] = df['text'].apply(preprocess_text)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = scores['pos'] if scores['pos'] > 0 else 0

    return sentiment


@app.route("/logs/", methods=["POST"])
def make_log():
    """
    Endpoint for getting a task by id
    """
    global dontwannachangedb
    global log
    body = json.loads(request.data)
    new_message = {"date":body.get("date"), "message":body.get("message")}
    log = body.get("message")
    dontwannachangedb = str(get_sentiment(new_message['message']))
    return jsonify({'sentiment':str(get_sentiment(new_message['message']))}),201
@app.route("/senti/")
def get_senti():
    """
    Endpoint for getting a task by id
    """
    return jsonify({'sentiment':dontwannachangedb}),201


@app.route("/goals/<int:goal_id>/", methods=["POST"])
def update_goal(goal_id):
    """
    Endpoint for getting a task by id
    """
    course = Goals.query.filter_by(id=goal_id).first() 
    body = json.loads(request.data)
    if course is None:
        return failure_response("Goal not found!")
    course.progress = body.get("progress")
    db.session.commit() 
    
    return jsonify(course.to_dict()), 201

import openai
import speech_recognition as sr
import pyttsx3
import os
import json
from openai import OpenAI
personality = "p.txt"
usewhisper = True

client = OpenAI()
# openAI set-up
openai.api_key = key
with open(personality, "r") as file:
    mode = file.read()
messages  = [{"role": "system", "content": f"{mode}"}]


# speech recognition set-up
r = sr.Recognizer()
mic = sr.Microphone()
r.dynamic_energy_threshold=False
r.energy_threshold = 400


def whisper(audio):
    print('iam here')
    with open('speech.wav','wb') as f:
        f.write(audio.get_wav_data())
    speech = open('speech.wav', 'rb')
    wcompletion = client.audio.transcriptions.create(
      model="whisper-1", 
      file=speech, 
      response_format="text"
    )
    print(wcompletion)
    # wcompletion = openai.Audio.transcribe(
    #     model = "whisper-1",
    #     file=speech
    # )
    user_input = wcompletion
    print(user_input)
    return user_input


def save_conversation(save_foldername):
    '''
    Checks the folder for previous conversations and will get the next suffix that has not been used yet.  
    It returns suffix number

    Args:
        save_foldername (str) : Takes in the path to save the conversation to.
    '''
    
    os.makedirs(save_foldername, exist_ok=True)

    base_filename = 'conversation'
    suffix = 0
    filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')

    while os.path.exists(filename):
        suffix += 1
        filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')

    with open(filename, 'w') as file:
        json.dump(messages, file, indent=4)

    return suffix

def save_inprogress(suffix, save_foldername):
    '''
    Uses the suffix number returned from save_conversation to continually update the 
    file for this instance of execution.  This is so that you can save the conversation 
    as you go so if it crashes, you don't lose to conversation.  Shouldn't be called
    from outside of the class.

    Args:
        suffix  :  Takes suffix count from save_conversation()
    '''
    os.makedirs(save_foldername, exist_ok=True)
    base_filename = 'conversation'
    filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')

    with open(filename, 'w') as file:
        json.dump(messages, file, indent=4)


# grab script location
script_dir = os.path.dirname(os.path.abspath(__file__))
foldername = "voice_assistant"
save_foldername = os.path.join(script_dir,f"conversations/{foldername}")
suffix = save_conversation(save_foldername)
#response = 'Absolutely, here are some tips on setting and achieving life goals:1. Identify your goals: Take some time to reflect and think about what is important to you. Write down your goals and be specific. Consider both short-term and long-term goals.2. Break them down: Once you have your goals, break them down into smaller, more manageable steps. This will make them less overwhelming and easier to achieve.3. Create a plan: Develop a plan of action to achieve your goals. This may include creating a schedule or timeline, seeking support from others, or learning new skills.4. Stay motivated: Set up a system of accountability to help keep you motivated. This may include sharing your goals with a friend or family member, tracking your progress, or rewarding yourself when you reach milestones.5. Be flexible: Remember that life is unpredictable, and its okay to adjust your goals and plans as needed. Stay open to new opportunities and be willing to change course if necessary.6. Take care of yourself: Achieving your goals requires hard work and dedication, but dont forget to take care of yourself along the way. Make time for self-care, rest, and relaxation. Remember, setting and achieving goals takes time and effort, but the rewards are worth it. By staying focused, motivated, and flexible, you can turn your dreams into reality.'
response = 'hi'
@app.route("/record")
def trigger_voicebot():
    """
    Endpoint for getting a task by id
    """
    global messages
    global response
    # body = json.loads(request.data)
    # loopguard = body.get("continue")
    # if(loopguard == 'reset'):
    #     messages  = [{"role": "system", "content": f"{mode}"}]

    with mic as source:
        print("\nListening...")
        r.adjust_for_ambient_noise(source, duration = 0.5)
        audio = r.listen(source)
        try:
            if usewhisper:
                user_input = whisper(audio)
            else:
                user_input = r.recognize_google(audio)
        except:
            print('')

    messages.append({"role" : "user", "content" : user_input})

    completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0301",
            messages=messages,
            temperature=0.8
        )    

    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    print(f"\n{response}\n")
    save_inprogress(suffix, save_foldername)
    
    # engine.say(f'{response}')
    # # engine.startLoop(False)
    # # # engine.iterate() must be called inside Server_Up.start()
    # # # Server_Up = threading.Thread(target = Comm_Connection)
    # # # Server_Up.start()
    # # engine.endLoop()
    # engine.runAndWait()
    # if engine._inLoop:
    #     engine.endLoop()


    
    return jsonify({}), 200
def text_to_speech(text):
    """
    Function to convert text to speech
    :param text: text
    :param gender: gender
    :return: None
    """
    engine = pyttsx3.init()

    # Setting up voice rate
    # engine.setProperty('rate', 125)

    # Setting up volume level  between 0 and 1
    # engine.setProperty('volume', 0.8)

    # Change voices: 0 for male and 1 for female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[14].id)

    
    engine.say(text)
    engine.runAndWait() 
    time.sleep(30)
    if engine._inLoop:
        engine.endLoop()

@app.route("/response/")
def get_response():
    """
    Endpoint for getting a task by id
    """
    global response
    text_to_speech(response)
    # pyttsx3 setup
    # engine = pyttsx3.init()
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[14].id) # 0 for male, 1 for female
    # engine.say(f'{response}')
    # engine.startLoop(False)
    # # engine.iterate() must be called inside Server_Up.start()
    # # Server_Up = threading.Thread(target = Comm_Connection)
    # # Server_Up.start()
    # engine.endLoop()
    # engine.runAndWait()
    # engine.stop()
    #engine.endLoop()
    # if engine._inLoop:
    #     engine.endLoop()
    return jsonify({}), 200

if __name__ == "__main__":
  app.run(debug=True)