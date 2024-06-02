import tkinter as tk
from movie import speech_to_text, extract_movie_titles, get_recommendations

class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation App")

        self.label = tk.Label(root, text="Welcome to Movie Recommendation App", font=("Helvetica", 16))
        self.label.pack(pady=20)

        self.speech_button = tk.Button(root, text="Speech", command=self.get_recommendation)
        self.speech_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.pack(pady=10)

    def get_recommendation(self):
        self.label.config(text="Please say something...")
        self.root.update()

        user_input = speech_to_text()

        if user_input:
            movie_titles = extract_movie_titles(user_input)
            if movie_titles:
                recommendations_text = "Extracted movie titles:\n"
                for title in movie_titles:
                    recommendations = get_recommendations(title)
                    recommendations_text += f"- {title}:\n{recommendations.head(10)}\n\n"
                self.label.config(text=recommendations_text)
            else:
                self.label.config(text="Please try again.")
        else:
            self.label.config(text="Please try again.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
