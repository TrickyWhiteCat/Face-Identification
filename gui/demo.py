import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import load_model
class WebcamApp:
    def __init__(self, window, window_title,model_path,model, video_source=0,threshold=0.7):
        # self.model_path = r'D:\20231\code\deep learning\project\pre train model\new world\firstbigtrain.pth'  
        self.threshold=threshold
        self.window = window
        self.window.title(window_title)
        self.model=load_model.LoadModel(model_path=model_path,model=model)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to take a picture
        self.btn_snapshot = ttk.Button(window, text="Check", command=self.snapshot)
        self.btn_snapshot.pack(pady=10)

        # Bind the 'q' key to quit the application
        self.window.bind('<q>', lambda event: self.window.destroy())

        self.update()
        self.window.mainloop()

    def snapshot(self):
        # Create the "database" folder if it doesn't exist
        database_folder = "database"
        if not os.path.exists(database_folder):
            os.makedirs(database_folder)
            print("Database folder created.")


        # Capture a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Open the second window to display the captured image
            image_window = tk.Toplevel(self.window)
            image_window.title("Captured Image")

            # Read the captured image and display it
            captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(captured_image)
            img_label = tk.Label(image_window, image=photo)
            img_label.image = photo
            img_label.pack()
            answer=load_model.getSamllestDistance(self.model,frame,self.threshold)
            if answer[0]==-1:
                text='unknown face'
            else:
                text=f'face recognite {answer[1]}'
            name=ttk.Label(image_window,text=text)
            name.pack()
            savePicture=ttk.Button(image_window,text='Save picture',command= lambda: self.saveProcess(image_window,answer[0],frame,answer[1],database_folder))
            savePicture.pack()
    def saveProcess(self,image_window,condition,frame,label,database_folder):
        if condition==-1:
            name=ttk.Entry(image_window)
            name.pack()
            saveButton=ttk.Button(image_window,text='save',command=lambda:  self.finalSave(name,frame,database_folder))
            saveButton.pack()
        else:
            person_folder = os.path.join(database_folder, label.replace('/', '_').replace('\\', '_'))
            # Find the largest number in the folder
            existing_numbers = [int(filename.split('.')[0]) for filename in os.listdir(person_folder) if filename.split('.')[0].isdigit()]

            if existing_numbers:

                largest_number = max(existing_numbers)
            else:
                largest_number = 0

            # Increment the largest number by 1
            new_number = largest_number + 1

            # Save the image with the new filename
            image_path = os.path.join(person_folder, f"{new_number}.png")
            captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            captured_image.save(image_path)

            print(f"Picture taken and saved as '{image_path}'")
    def finalSave(self,label,frame,database_folder):
        if not os.path.exists(label.get()):
            os.makedirs(f'{database_folder}\\{label.get()}')
            image_path = os.path.join(database_folder, f"{label.get()}\\0.png")
            captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            captured_image.save(image_path)
            print(f"Picture taken and saved as '{image_path}'")
            

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Convert the frame to RGB format and display in the tkinter window
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.window.photo = photo

        # Call the update method after 10 milliseconds
        self.window.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
if __name__=='__main__':
    model_path = r'load\firstbigtrain.pth'  
    model=load_model.VGG16_NET()
    # Create a window and pass it to the WebcamApp class
    root = tk.Tk()
    app = WebcamApp(root, "Face Identification App",model_path=model_path,model=model,threshold=0.5)
