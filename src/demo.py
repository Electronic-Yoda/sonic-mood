import tkinter as tk
from tkinter import filedialog
import torchaudio, torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.insert(0, BASE_DIR)
from utils.constants import Emotions
from models.audio_cnn import AudioCNN, ModAudioCNN8
from utils.transform import RawAudioToMelspecTransform


# Global variables
reddish = '#6E4137'
yellowish = '#D1CFA6'
orangeish = '#929850'
blueish = '#356885'
lighterblueish = '#1F3A57'
darkgreyish='#797D7F'
darkish = '#17202A'
greyish = '#A69E99'
lightgreen = '#85FF84'
greenblue = '#4986A3'
lightyellow = '#E0DF8A'
lightblue1 = '#8FA5BF'
lightblue2 = '#8FB0BF'
almostwhite = '#fafafa'
window_width = 750
window_height = 400


class SonicMoodUI:

    def __init__(self, master, net=None):
        self.master = master
        master.title("Sonic Mood")
        master.configure(bg=blueish)
        master.geometry(f"{window_width}x{window_height}")
        master.iconbitmap(default='')

        self.add_audio_button = tk.Button(master, text='Add Audio', command=self.load_audio,
            background=lightblue2, foreground='white', font=('Fixedsys', 14), borderwidth=3)
        # self.add_audio_button.pack()

        self.predict_emotion_button = tk.Button(master, text='Predict Emotion', command=self.predict_emotion,
            background=lightblue2, foreground='white', font=('Fixedsys', 14), borderwidth=3)
        # self.predict_emotion_button.pack()

        self.result_label = tk.Label(master, text='Predicted Emotion Will Appear Here', font=('Fixedsys', 14), bg='black', foreground='yellow', border=3)
        # self.result_label.pack()
        # Create a canvas for the plot placeholder
        self.plot_placeholder = tk.Canvas(master, width=400, height=200, bg='grey')

        self.waveform = None
        self.canvas = None


        # Place the widgets on the window
        leftx = 50
        self.buttonwidth = 200
        self.buttonheight = 50
        rightx = leftx + self.buttonwidth + 50
        # align left side of widget with leftx
        self.add_audio_button.place(x=leftx, y=100, width=self.buttonwidth, height=self.buttonheight * 2)
        self.predict_emotion_button.place(x=leftx, y=300, width=self.buttonwidth, height=self.buttonheight)
        self.result_label.place(x=rightx, y=300, width=400, height=self.buttonheight)
        self.rightx = rightx

        
        self.plot_placeholder.place(x=self.rightx, y=50, width=400, height=200)

        # Add some placeholder text
        self.plot_placeholder.create_text(200, 100, text="Mel Spectrogram will appear here", font=('Fixedsys', 14))

        self.audio_transform = RawAudioToMelspecTransform(test_mode=True)
        self.net = net

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)


    # Function to load .wav file
    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[('Audio Files', '*.wav')])
        if file_path:
            waveform, sample_rate = torchaudio.load(file_path)
            # Your processing function goes here
            input = self.audio_transform(waveform, sample_rate)

            fig, ax = plt.subplots(figsize=(4, 2))
            # Add a border to the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('grey')
                spine.set_linewidth(2)
            ax.imshow(input[0].numpy(), cmap='hot', origin='lower', aspect='auto')
            ax.set_title('Mel Spectrogram')
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Bands')
            ax.set_xlim([0, input.shape[2]])

            # Remove the placeholder canvas if it exists
            if self.plot_placeholder:
                self.plot_placeholder.destroy()

            # Remove previous canvas if it exists
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.master)
            self.canvas.draw()
            canvas_widget = self.canvas.get_tk_widget()

            # Place the canvas widget
            canvas_widget.place(x=self.rightx, y=50, width=fig.get_figwidth()*100, height=fig.get_figheight()*100)

            # Create a frame with a border
            self.plot_frame = tk.Frame(self.master, borderwidth=2, relief="solid", bg='grey')
            self.plot_frame.place(x=self.rightx, y=50, width=400, height=200)

            # Place the canvas inside the frame
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            canvas_widget = self.canvas.get_tk_widget()

            # Place the canvas widget inside the frame
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # reset result label
            self.result_label.config(text='Predicted Emotion Will Appear Here')

            self.waveform = input
            

    # Function to predict emotion
    def predict_emotion(self):
        if self.waveform is not None:
            print('predicting emotion')
            
            outputs = self.net(self.waveform)
            pred = outputs.max(1, keepdim=True)[1]
            predicted_emotion = Emotions.from_index(pred.item())
            # print the predicted emotion on the Tkinter GUI
            self.result_label.config(text=str(predicted_emotion))
            
        else:
            self.result_label.config(text="Please load an audio file first")

    # Function to close all Matplotlib figures
    def on_closing(self):
        plt.close('all')
        self.master.destroy()

if __name__ == '__main__':

    root = tk.Tk()
    net = ModAudioCNN8()
    model_state_path = os.path.join(BASE_DIR, 'src', 'models', 'model_state', 'ModAudioCNN8unbalanced_bs4_lr1e-05_epoch29')
    net.load_state_dict(torch.load(model_state_path))
    app = SonicMoodUI(root, net=net)
    root.mainloop()
