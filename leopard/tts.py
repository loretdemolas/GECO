import pyttsx3


class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


# For testing
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Hello, how are you?")

