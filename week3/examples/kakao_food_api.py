import os
import requests
from PIL import Image, ImageDraw, ImageFont

def detection(filename):
    food_url = 'https://dapi.kakao.com/food-affiliate/v2/detect.json'
    app_key = '442afbe060465f18c8fd877e67baf3e3'
    headers = {'Authorization': f'KakaoAK {app_key}'}
    files = {'imgfile': open(filename, 'rb')}
    
    try:
        response = requests.post(food_url, headers = headers, files = files)
        if response.status_code == 200:
            result = response.json()
            image = Image.open(filename)
            draw = ImageDraw.Draw(image)
            for i, box in enumerate(result['results']['food']):
                x = int(box['x'])
                y = int(box['y'])
                w = int(box['w'])
                h = int(box['h'])
                draw.rectangle([(x, y), (x + w, y + h)], fill = None,  outline = (255,0,0,255))
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 30, )
                draw.text((x + 5, y + 5), result['results']['food'][i]['class_info'][0]['food_name'], font = font)
            image.show()
            return image
        else:
            print(response)
    except Exception as e:
        print(f'Error: {e}')
        
detection('food3.jpg')