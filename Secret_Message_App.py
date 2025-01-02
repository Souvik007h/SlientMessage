! pip install opencv-python
import streamlit as st
import cv2
import uuid
import numpy as np
import string
import random
import os
# For text message input
def embed_text_message(cover_image, secret_message):
    def create_actual_secret(secret_message):
        def generate_unique_id():
            # Define the pool of characters and digits
            letters = string.ascii_letters
            digits = string.digits

            # Ensure at least two digits are included
            id_components = random.sample(digits, 2)  # Pick 2 unique digits

            # Fill the remaining spots with letters
            id_components += random.choices(letters + digits, k=4)  # Random mix of letters and digits

            # Shuffle to ensure randomness
            random.shuffle(id_components)

            # Build the ID ensuring no two identical characters are adjacent
            unique_id = [id_components.pop(0)]  # Start with the first character
            while id_components:
                next_char = id_components.pop(random.randint(0, len(id_components) - 1))
                if next_char != unique_id[-1]:  # Ensure no two adjacent are identical
                    unique_id.append(next_char)
                else:
                    # If the character is the same as the last, push it back and retry
                    id_components.append(next_char)
                    random.shuffle(id_components)  # Shuffle to prevent infinite loops

            return ''.join(unique_id)

        #secret message constuction for embedding
        secret = secret_message
        uid = generate_unique_id()
        file_extension = ".txt"
        message =  uid +"|"+ file_extension +"|"+ secret+"|"
        binary__message = ''.join(format(ord(char), '09b') for char in message)
        #padding
        if len(binary__message) % 27 != 0:
            padding_needed = 27 - (len(binary__message) % 27)
            binary__message += '0' * padding_needed

        #Adding end symbol to message
        END_SYMBOL = "|\|"
        END_SYMBOL_BINARY = ''.join(format(ord(char), '09b') for char in END_SYMBOL)

        total_message =  binary__message + END_SYMBOL_BINARY 
        
        return total_message, uid


    def resize_image_based_on_secret_length(img, secret_length):
        # List of image sizes and their corresponding thresholds
        size_thresholds = [
            (360, 360, 388800),
            (512, 512, 780300),
            (720, 720, 1555200),
        ]

        for width, height, threshold in size_thresholds:
            if secret_length < threshold:
                resized_image = cv2.resize(img, (width, height))
                return resized_image

        # If secret_length exceeds all thresholds, resize to the largest size
        largest_size = size_thresholds[-1]
        return cv2.resize(img, (largest_size[0], largest_size[1]))

    def Embedding(mat, secret):
        BLTM = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ])

        BUTM = np.array([
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1]
        ])
        sec_1 = secret[:9]
        sec_2 = secret[9:18]
        sec_3 = secret[18:27]

        # Convert to 3x3 matrices
        mat_1 = np.array([int(char) for char in sec_1]).reshape(3,3)
        mat_2 = np.array([int(char) for char in sec_2]).reshape(3,3)
        mat_3 = np.array([int(char) for char in sec_3]).reshape(3,3)

        mat_1_t = np.bitwise_xor(mat_1, BLTM).T
        mat_2_t = np.bitwise_xor(mat_2, BUTM).T
        mat_3_t = np.bitwise_xor(mat_3, BLTM).T

        stego = mat.copy()

        row_map = {
        1: [0,0],
        2: [1,2],
        3: [2,0],
        4: [0,2],
        5: [1,0],
        6: [2,2],
        7: [0,1],
        8: [1,1],
        9: [2,1]
        }

        def get_lsb(value):
            li = []
            for i in value:
                li.append(int(format(i, '08b')[-1]))
            return li

        def mod_2(matrix):
            return matrix % 2

        # Step 8 & 9: Calculate z values for each row
        def calculate_z(V):
            V = np.array(V).reshape(-1, 1)
            return mod_2(np.matmul(BLTM, V)).T

        # Step 10: Calculate delta values
        def get_m_secret(row_num):
            if row_num in [1, 2, 3]:
                return mat_1_t[row_num - 1]
            elif row_num in [4, 5, 6]:
                return mat_2_t[row_num - 4]
            else:
                return mat_3_t[row_num - 7]

        # Step 11: Delta to v_n mapping
        delta_to_v_n = {
            (0, 0, 0): (0, 0, 0),
            (0, 0, 1): (0, 0, 1),
            (0, 1, 0): (0, 1, 1),
            (0, 1, 1): (0, 1, 0),
            (1, 0, 0): (1, 1, 0),
            (1, 0, 1): (1, 1, 1),
            (1, 1, 0): (1, 0, 1),
            (1, 1, 1): (1, 0, 0),
        }

        # Step 12: Calculate v_s values
        def calculate_v_s(V, z, row_num):
            m_secret = get_m_secret(row_num)
            delta = np.bitwise_xor(z.flatten(), m_secret)
            v_n = delta_to_v_n[tuple(delta)]
            return np.bitwise_xor(V, v_n)

        # Step 13: Modify pixel values with v_s
        def modify_pixel_with_v_s(pixel, bit):
                # Ensure v_s_bit is 0 or 1
                binary = format(pixel, '08b')
                tem = int(binary[-1])
                if(tem == bit):
                    return pixel
                elif(tem == 0 and bit == 1):
                    return pixel + 1
                else:
                    return pixel - 1

        # Process C1 rows (1,3,5)
        for row_num in range(1,10):
            V = get_lsb(mat[row_map[row_num][0]][row_map[row_num][1]])
            z = calculate_z(V)
            v_s = calculate_v_s(V, z, row_num)
            original_row = mat[row_map[row_num][0]][row_map[row_num][1]]
            modified_row = [modify_pixel_with_v_s(pixel, v_s_bit)
                        for pixel, v_s_bit in zip(original_row, v_s)]

            stego[row_map[row_num][0]][row_map[row_num][1]] = modified_row
        return stego

    def perfrom_embedding(img,secret):
        stego_image = img.copy()
        count = 0
        found = False
        for i in range(0, img.shape[0] - 2, 3):
            for j in range(0, img.shape[1] - 2, 3):
                block = img[i:i+3, j:j+3]
                stego_image[i:i+3, j:j+3] = Embedding(block, secret[count])
                if(secret[count] == '001111100001011100001111100'):
                    found = True
                    break
                count+=1
            if(found):
                break
        
        return stego_image

    # Total message
    total_message, uid = create_actual_secret(secret_message)
    # Create a secret in a list type
    secret = [total_message[i:i+27] for i in range(0, len(total_message), 27)]
            
        
    image = resize_image_based_on_secret_length(cover_image, len(total_message))

    stego_image = perfrom_embedding(image, secret)  # Replace with your actual embedding logic
    return stego_image, uid


# Define extraction function
def extract_secret_message(stego_image, unique_id):
    def mod_2(matrix):
        return matrix % 2


    def get_lsb(value):
        li = []
        for i in value:
            li.append(int(format(i, '08b')[-1]))
        return li


    def extract_secret_from_block(stego):
        BLTM = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ])

        BUTM = np.array([
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1]
        ])
        row_map = {
            1: [0,0],
            2: [1,2],
            3: [2,0],
            4: [0,2],
            5: [1,0],
            6: [2,2],
            7: [0,1],
            8: [1,1],
            9: [2,1]
        }
        # Extract rows for secret parts
        result = []
        for i in range(1,10):
            v_1 = np.array(get_lsb(stego[row_map[i][0]][row_map[i][1]])).reshape(-1, 1)
            m = mod_2(np.matmul(BLTM, v_1)).T
            result.append(m)

        # print(result)

        matrix_1 = np.vstack(result[0:3]).T
        matrix_2 = np.vstack(result[3:6]).T
        matrix_3 = np.vstack(result[6:9]).T
        sec_1 = np.bitwise_xor(BLTM, matrix_1).reshape(1, 9)
        sec_2 = np.bitwise_xor(BUTM, matrix_2).reshape(1, 9)
        sec_3 = np.bitwise_xor(BLTM, matrix_3).reshape(1, 9)

        # Convert to string format
        sec_1_str = ''.join(map(str, sec_1.flatten()))
        sec_2_str = ''.join(map(str, sec_2.flatten()))
        sec_3_str = ''.join(map(str, sec_3.flatten()))

        return sec_1_str + sec_2_str + sec_3_str


    def extraction(image):
        d1_image = image
        secret_string = ""
        found = False
        for i in range(0, d1_image.shape[0] - 2, 3):
            for j in range(0, d1_image.shape[1] - 2, 3):
                block = d1_image[i:i+3, j:j+3]
                s = extract_secret_from_block(block)
                # print("Secret message",s)
                secret_string+=s
                if(s == "001111100001011100001111100"):
                    found = True
                    break
            if(found):
                break
        return secret_string

    def extract_info(secret_string):
        elements = []
        original_string = ''.join(chr(int(secret_string[i:i+9], 2)) for i in range(0, len(secret_string), 9))
        for i in original_string.split("|"):
            elements.append(str(i))
        
        return elements[0], elements[1], elements[2]

    secret = extraction(stego_image)

    uid, file_extension, messages = extract_info(secret)

    if(uid == unique_id_input):
        return messages
    else:
        return "Unique ID does not match, extraction failed"
    

st.title("Silent Message App")
st.write("### Make your message secret")
st.write("#### Choose between embedding or extracting a secret message!")

# Mode Selection
mode = st.radio("Select Mode:", ["Embed", "Extract"])

if mode == "Embed":
    st.header("Embed a Secret Message")

    # Cover Image Upload
    cover_image_file = st.file_uploader("Upload Cover Image (Max: 10 MB)", type=["jpg", "jpeg", "png"])
    if cover_image_file:
        # Process the cover image
        cover_image = cv2.imdecode(np.frombuffer(cover_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert to RGB for display (since OpenCV loads in BGR format)
        cover_image_rgb = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)

        # Display the cover image
        st.image(cover_image_rgb, caption="Cover Image", use_container_width=True)

    # Secret Message Input
    st.write("### Provide Your Secret Message")
    secret_message = st.text_area("Write your secret message here: (Max: 194,000 characters)")

    # Generate Stego Image
    if cover_image_file:
        if st.button("Generate Stego Image"):
            if secret_message:
                # Call the embedding function for the written message
                stego_image, unique_id = embed_text_message(cover_image, secret_message)

                # Save the stego image as a PNG file using the unique ID
                stego_image_path = "stego_image.png"
                cv2.imwrite(stego_image_path, stego_image)

                # Display the unique ID
                st.success(f"Unique ID: {unique_id}")
                st.code(unique_id)
                st.write("Make sure you copy this unique ID for secret message extraction")

                # Provide a download button for the stego image
                download_clicked = st.download_button(
                    "Download Stego Image",
                    data=open(stego_image_path, "rb"),
                    file_name=f"stego_image.png"
                )

                # Handle page reload after download
                if download_clicked:
                    # Remove the temporary file after download
                    os.remove(stego_image_path)
                    # Clear inputs by reloading the page
                    st.success("The page will reload in 3 seconds...")
                    st.session_state["reload_trigger"] = True
                    st.experimental_rerun()


                
elif mode == "Extract":
    st.header("Extract a Secret Message")

    # Stego Image Upload
    stego_image_file = st.file_uploader("Upload Stego Image (Max: 10 MB)", type=["jpg", "jpeg", "png"])

    # Unique ID Input
    unique_id_input = st.text_input("Enter Unique ID:")

    if st.button("Extract Secret Message"):
        if stego_image_file and unique_id_input:
            # Process the stego image
            stego_image = cv2.imdecode(np.frombuffer(stego_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Extract the secret message
            secret_message = extract_secret_message(stego_image, unique_id_input)

            # Display the extracted secret message
            st.write("### Extracted Secret Message:")
            st.success(secret_message)
        else:
            st.error("Please upload a stego image and enter the unique ID!")
