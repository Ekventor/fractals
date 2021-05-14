from flask import Flask, request, json
from vk_api import VkUpload
import vk_api
import numpy as np
from matplotlib import pyplot as plt
import traceback
from numba import jit


TOKEN = "access_token"

vk = vk_api.VkApi(token=TOKEN)
vk._auth_token()

upload = VkUpload(vk)

ids = []


def mandelbrot(formula, author_id, peer_id, is_club=False, filename="fractal.png"):
    calculate_z = jit(nopython=True, parallel=True)(eval(f"lambda z, c, p, q, k: {formula}"))

    def get_fractal(pmin, pmax, ppoints, qmin, qmax, qpoints, max_iterations, infinity_border):
        image = np.zeros((ppoints, qpoints))
        p, q = np.mgrid[pmin:pmax:(ppoints * 1j), qmin:qmax:(qpoints * 1j)]
        c = p + 1j * q
        z = np.zeros_like(c)

        for k in range(max_iterations):
            z = calculate_z(z, c, p, q, k)
            mask = (np.abs(z) > infinity_border) & (image == 0)
            image[mask] = k
            z[mask] = np.nan

        return -image.T

    image = get_fractal(-2.5, 1.5, 1000, -2, 2, 1000, 200, 20)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap="flag", interpolation="none")

    fig = plt.gcf()
    fig.set_size_inches(20, 20)

    plt.axis("off")
    plt.savefig(filename, format="png", bbox_inches="tight")

    photo_list = upload.photo_messages(filename)
    attachment = ",".join("photo{owner_id}_{id}".format(**item) for item in photo_list)

    vk.method("messages.send", {"peer_id": peer_id,
                                "message": f'[{"id" if not is_club else "public"}{author_id}|Автор]\nФормула: {formula}',
                                "attachment": attachment, 
                                "random_id": 0})


app = Flask(__name__)


@app.route("/", methods=["POST"])
def main():
    try:
        try:
            data = json.loads(request.data.decode("UTF-8"))
        except json.decoder.JSONDecodeError:
            return "ok"

        if data["type"] == "confirmation":
            return "callback_string"

        elif data["type"] == "message_new":
            user_id = data["object"]["message"]["from_id"]

            if user_id < 0:
                is_club = True
                user_id *= -1
            else:
                is_club = False

            peer_id = data["object"]["message"]["peer_id"]
            message = data["object"]["message"]["text"]
            message_id = data["object"]["message"]["conversation_message_id"]

            if "/фрактал" in message:
                if f"{peer_id}: {message_id}" not in ids:
                    if "club" in message:
                        formula = message.split(" ", 2)[2]
                    else:
                        formula = message.split(" ", 1)[1]

                    ids.append(f"{peer_id}: {message_id}")
                    mandelbrot(formula, user_id, peer_id, is_club)

    except:
        pass

    return "ok"
