import os

def get_image_paths_and_names(folder_path):
    image_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(folder_path, filename)
            image_list.append((file_path, filename))

    return image_list

def get_person_folders(directory_path):
    person_folders = []

    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        
        if os.path.isdir(folder_path):
            list_person=get_image_paths_and_names(folder_path)
            for link,person in list_person:
                person_folders.append((link, folder_name))

    return person_folders
if __name__=='__main__':
    # Example usage:
    main_directory = r"database"
    folders_info = get_person_folders(main_directory)

    for folder_path, person_name in folders_info:
        print(f"Folder Path: {folder_path}, Person Name: {person_name}")
        # Use folder_path and person_name as needed in your application
