# Rubik's Cube Solver (Computer Vision + Deep Learning)

A CNN reads the colours of a Rubik's cube from photos or a webcam, and the Kociemba algorithm solves it.
The web app is built with Streamlit; the model is trained in Google Colab.

```
photo -> find the cube face (OpenCV) -> CNN reads 9 sticker colours -> 54-sticker state -> Kociemba solver -> moves
```

**Which part is deep learning?** The CNN (`train_eval.py: build_face_cnn`) trained from scratch.
**Which part is not?** Finding the face (OpenCV) and solving (Kociemba two-phase algorithm). Solving a cube is a search
problem; learned solvers exist but need far more compute than free Colab provides.

## Files

| File | Purpose |
|---|---|
| `rubiks_cube_training.ipynb` | Colab notebook: data, training, evaluation, baselines, saving the model |
| `cube_utils.py` | Face detection, cube simulator, colour decoding, validation, solver, drawing |
| `app.py` | Streamlit web app (upload + webcam) |
| `requirements.txt` | Packages for the app |
| `face_cnn.keras` | Created by the notebook (not included) |

(`synth.py` and `train_eval.py` are written by the notebook itself and are only needed for training.)

## How to run

1. Open `rubiks_cube_training.ipynb` in Google Colab, set the runtime to **T4 GPU**, run all cells, download `face_cnn.keras`.
2. Put `app.py`, `cube_utils.py`, `requirements.txt` and `face_cnn.keras` in one folder.
3. `pip install -r requirements.txt`, then `streamlit run app.py`.

Without `face_cnn.keras` the app still runs using the "centre-nearest colour" reader (no training needed), but it is less accurate.
Use **Load demo cube** in the sidebar to try the solver without photos.

## How to scan the cube

Pick one face as F (towards the camera) and one as U (on top). Then scan in any order:
F as it is; R, B, L by turning the whole cube so the next face on the right comes to the camera (U stays on top);
U by tilting the top towards the camera; D by tilting the bottom towards the camera. The app shows these instructions per face.
Use a plain background and good light, and let the face fill most of the picture.

## Deploying

* **Streamlit Community Cloud** or **Hugging Face Spaces (Streamlit)**: push the four files above to a repository and deploy.
  Webcam access works because these sites use HTTPS.
* **Google Colab**: possible through a tunnel (last notebook cell), but the session is temporary.
* The webcam uses `st.camera_input` (take a photo per face). A continuous live video feed would need the extra package
  `streamlit-webrtc`; it is not included.

## How the colour reading is evaluated

The training data are synthetic faces whose sticker colours are sampled from a public dataset of real stickers
(HSV values under 5 lighting conditions, GitHub: eyeeco/Rubick-s-Cube-Dataset). The test faces use a lighting condition
that was left out of training. The notebook then evaluates whole cubes (all 54 stickers correct, then solved by Kociemba)
and compares the CNN with a Random Forest on HSV values and a no-training baseline.
Synthetic results are only a guide: measure on photos of your own cube (notebook step 10).

## Limitations

* The public colour dataset covers only two cubes, so a cube with different colours may be read worse. Step 11 of the notebook
  fine-tunes on your own labelled photos.
* Red vs orange and white vs yellow are the hardest pairs. The app flags low-confidence stickers and lets you correct any sticker
  before solving.
* One wrongly read sticker makes the cube unsolvable; the app then shows which colour counts are wrong.
