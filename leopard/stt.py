import speech_recognition as sr


class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.responded = False

    def listen(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = self.recognizer.listen(source)
            print("Listening...")
        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
            return ""
        except sr.RequestError as e:
            print("Sorry, there was an error processing your request. Please try again later.")
            return ""


# For testing
if __name__ == "__main__":
    stt = SpeechToText()
    print(stt.listen())
