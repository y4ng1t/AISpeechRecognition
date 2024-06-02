import tkinter as tk
from tkinter import messagebox
import threading
import movie  # Ensure movie.py is in the same directory and contains the necessary functions

# Function to handle speech recognition
def recognize_speech():
    # Disable the speech button to prevent multiple clicks
    speech_button.config(state=tk.DISABLED)
    # Update the recommendation label to prompt the user
    recommendation_label.config(text="Please say something...")
    # Call the speech_to_text function from movie.py
    user_input = movie.speech_to_text()
    if user_input:
        # Extract movie titles from the speech
        movie_titles = movie.extract_movie_titles(user_input)
        if movie_titles:
            # Get recommendations for the first movie title found
            recommendations = movie.get_recommendations(movie_titles[0])
            # Update the recommendation label with the first recommendation
            recommendation_label.config(text=recommendations.iloc[0])
        else:
            # If no movie titles were found, prompt the user to try again
            recommendation_label.config(text="Please try again")
    else:
        # If speech_to_text failed, prompt the user to try again
        recommendation_label.config(text="Please try again")
    # Re-enable the speech button
    speech_button.config(state=tk.NORMAL)

# Function to run speech recognition in a separate thread
def start_speech_recognition():
    threading.Thread(target=recognize_speech).start()

# Function to exit the application
def exit_application():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Movie Recommendation")

# Create a label for recommendations
recommendation_label = tk.Label(root, text="Recommendation movie will appear here", wraplength=300)
recommendation_label.pack(pady=10)

# Create a speech button
speech_button = tk.Button(root, text="Speech", command=start_speech_recognition)
speech_button.pack(pady=5)

# Create an exit button
exit_button = tk.Button(root, text="Exit", command=exit_application)
exit_button.pack(pady=5)

# Run the main loop
root.mainloop()
