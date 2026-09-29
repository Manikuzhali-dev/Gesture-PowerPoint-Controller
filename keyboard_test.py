import pyautogui
import time

print("Starting in 3 seconds...")
time.sleep(3)

print("Pressing RIGHT")
pyautogui.press("right")

time.sleep(1)

print("Pressing LEFT")
pyautogui.press("left")

print("Done!")