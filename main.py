import base64
import streamlit as st
import easyocr
from googletrans import Translator
from gtts import gTTS
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image

translator = Translator()


def display_text(bounds):
    text = []
    for x in bounds:
        t = x[1]
        text.append(t)
    text = ' '.join(text)
    return text


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack('background2.png')


image = Image.open('log.png')
st.sidebar.image(image, width=320)

options = st.sidebar.selectbox("Choose one of the following:",("Home","Scan image to pdf"))

def get_options(options):
    st.sidebar.write("\n")
    if (options == "Scan image to pdf"):
        scan_image()

    elif (options == "Home"):
        home_page()
def home_page():
    st.sidebar.title('Language Selection Menu')
    st.sidebar.subheader('Select...')
    src = st.sidebar.selectbox("From Language", ['English', 'Turkish', 'Arabic', 'Hungarian', 'Japanese', 'Italian', 'Dutch'])

    st.sidebar.subheader('Select...')
    destination = st.sidebar.selectbox("To Language", ['Dutch','Italian', 'Japanese','Hungarian', 'Arabic', 'Turkish', 'English'])

    st.sidebar.subheader("Enter Text")
    area = st.sidebar.text_area("Auto Detection Enabled", "")

    helper = {'Hungarian': 'hu', 'Italian': 'it', 'English': 'en', 'Arabic': 'ar', 'Japanese': 'ja', 'Dutch': 'nl', 'Turkish': 'tr'}
    dst = helper[destination]
    source = helper[src]

    if st.sidebar.button("Translate!"):
        if len(area) != 0:
            sour = translator.detect(area).lang
            answer = translator.translate(area, src=f'{sour}', dest=f'{dst}').text
            # st.sidebar.text('Answer')
            st.sidebar.text_area("Answer", answer)
            st.balloons()
        else:
            st.sidebar.subheader('Enter Text!')

    html_temp = """
      <div style="color:black;font-weight: bold;text-align:center;font-family:verdana;font-size:300%;">
      <span style="color: blue">TALAN'</span>AIR
      <br> <span style="color: blue">T</span>r<span style="color: blue">A</span>ve<span style="color: blue">L</span> 
      <span style="color: blue">AN</span>ywhere
      </div>
      </div>
      """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader('Upload the Image')

    image_file = st.file_uploader("", type=['jpg', 'png', 'jpeg', 'JPG'])

    if st.button("Convert"):

        if image_file is not None:
            img = Image.open(image_file)
            img = np.array(img)

            st.subheader('Image you Uploaded...')
            st.image(image_file, width=450)

            if src == 'English':
                with st.spinner('Extracting Text from given Image'):
                    eng_reader = easyocr.Reader(['en'])
                    detected_text = eng_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)

            if src == 'Turkish':
                with st.spinner('Extracting Text from given Image'):
                    eng_reader = easyocr.Reader(['tr'])
                    detected_text = eng_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)

            if src == 'Japanese':
                with st.spinner('Extracting Text from given Image'):
                    eng_reader = easyocr.Reader(['ja'])
                    detected_text = eng_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)

            if src == 'Dutch':
                with st.spinner('Extracting Text from given Image'):
                    eng_reader = easyocr.Reader(['nl'])
                    detected_text = eng_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)


            elif src == 'Italian':
                with st.spinner('Extracting Text from given Image'):
                    swahili_reader = easyocr.Reader(['it'])
                    detected_text = swahili_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)


            elif src == 'Hungarian':
                with st.spinner('Extracting Text from given Image'):
                    afrikaans_reader = easyocr.Reader(['hu'])
                    detected_text = afrikaans_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)


            elif src == 'Arabic':
                with st.spinner('Extracting Text from given Image...'):
                    arabic_reader = easyocr.Reader(['ar'])
                    detected_text = arabic_reader.readtext(img)
                st.subheader('Extracted text is ...')
                text = display_text(detected_text)
                st.write(text)
            st.write('')
            ta_tts = gTTS(text, lang=f'{source}')
            ta_tts.save('trans.mp3')
            st.audio('trans.mp3', format='audio/mp3')

            with st.spinner('Translating Text...'):
                result = translator.translate(text, src=f'{source}', dest=f'{dst}').text
            st.subheader("Translated Text is ...")
            st.write(result)

            st.write('')
            st.header('Generated Audio')

            with st.spinner('Generating Audio ...'):
                ta_tts2 = gTTS(result, lang=f'{dst}')
                ta_tts2.save('trans2.mp3')
            st.audio('trans2.mp3', format='audio/mp3')
            st.balloons()


        else:
            st.subheader('Image not found! Please Upload an Image.')

def scan_image():
    html_temp = """
      <div style="color:black;font-weight: bold;text-align:center;font-family:verdana;font-size:300%;">
      <span style="color: blue">TALAN'</span>AIR
      <br> <span style="color: blue">T</span>r<span style="color: blue">A</span>ve<span style="color: blue">L</span> 
      <span style="color: blue">AN</span>ywhere
      </div>
      </div>
      """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader('Upload the Image')
    image_file = st.file_uploader("", type=['jpg', 'png', 'jpeg', 'JPG'])
    if st.button("Convert"):
        img = Image.open(image_file)
        img = np.array(img)
        output = Image.fromarray(img)
        output.save('test.jpg')
        st.subheader('Image you Uploaded...')
        st.image(image_file, width=450)

        def opencv_resize(image, ratio):
            width = int(image.shape[1] * ratio)
            height = int(image.shape[0] * ratio)
            dim = (width, height)
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        def plot_rgb(image):
            plt.figure(figsize=(16, 10))
            return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        def plot_gray(image):
            plt.figure(figsize=(16, 10))
            return plt.imshow(image, cmap='Greys_r')

        # approximate the contour by a more primitive polygon shape
        def approximate_contour(contour):
            peri = cv2.arcLength(contour, True)
            return cv2.approxPolyDP(contour, 0.032 * peri, True)

        def get_receipt_contour(contours):
            # loop over the contours
            for c in contours:
                approx = approximate_contour(c)
                # if our approximated contour has four points, we can assume it is receipt's rectangle
                if len(approx) == 4:
                    return approx

        def contour_to_rect(contour, resize_ratio):
            pts = contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            # top-left point has the smallest sum
            # bottom-right has the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            # compute the difference between the points:
            # the top-right will have the minumum difference
            # the bottom-left will have the maximum difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect / resize_ratio

        def wrap_perspective(img, rect):
            # unpack rectangle points: top left, top right, bottom right, bottom left
            (tl, tr, br, bl) = rect
            # compute the width of the new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            # compute the height of the new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            # take the maximum of the width and height values to reach
            # our final dimensions
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            # destination points which will be used to map the screen to a "scanned" view
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            # calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst)
            # warp the perspective to grab the screen
            return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        def bw_scanner(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            T = threshold_local(gray, 21, offset=5, method="gaussian")
            return (gray > T).astype("uint8") * 255

        def Receipt_OCR(file_name):
            img = Image.open(file_name)
            image = cv2.imread(file_name)
            # Downscale image as finding receipt contour is more efficient on a small image
            resize_ratio = 500 / image.shape[0]
            original = image.copy()
            image = opencv_resize(image, resize_ratio)
            # Convert to grayscale for further processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Get rid of noise with Gaussian Blur filter
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Get rid of noise with Gaussian Blur filter
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Detect white regions
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilated = cv2.dilate(blurred, rectKernel)
            edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
            # Detect all contours in Canny-edged image
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
            # Get 10 largest contours
            largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0, 255, 0), 3)
            receipt_contour = get_receipt_contour(largest_contours)
            receipt_contour = get_receipt_contour(largest_contours)
            image_with_receipt_contour = cv2.drawContours(image.copy(), [receipt_contour], -1, (0, 255, 0), 2)
            scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))
            result = bw_scanner(scanned)
            output = Image.fromarray(result)
            output.save('result.png')

            return result

        Receipt_OCR("test.jpg")
        imgsc = Image.open('result.png')
        st.subheader('Image you Uploaded...')
        st.image(imgsc, width=450)
        im1 = imgsc.convert('RGB')
        im1.save('result.pdf')

        def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<button style="background-color:gray"><a style="text-decoration:none; color:white" href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a></button>'
            return href

        st.sidebar.markdown(get_binary_file_downloader_html('result.pdf', 'pdf'), unsafe_allow_html=True)

get_options(options)
