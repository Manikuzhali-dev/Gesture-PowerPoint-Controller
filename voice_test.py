import speech_recognition as sr
import pyautogui

recognizer = sr.Recognizer()

print("🎙️ Voice Controller Started")
print("Say: next slide | previous slide | resume | exit")
print("-" * 50)

while True:
    try:
        with sr.Microphone() as source:
            print("\n🎙️ Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=3
            )

        command = recognizer.recognize_google(audio).lower()
        print(f"🗣️ You said: {command}")

        if "next slide" in command:
            print("➡️ Next Slide")
            pyautogui.press("right")

        elif "previous slide" in command:
            print("⬅️ Previous Slide")
            pyautogui.press("left")

        elif "resume" in command:
            print("▶️ Resume")
            pyautogui.press("space")

        elif "exit" in command:
            print("🛑 Voice Controller Stopped")
            break

        else:
            print("⚠️ Command not recognized")

    except sr.WaitTimeoutError:
        print("⏱️ No speech detected.")

    except sr.UnknownValueError:
        print("❌ Couldn't understand.")

    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        break