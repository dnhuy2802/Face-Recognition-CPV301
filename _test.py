from app.logic.eigenfaces import MeanFaces
from app.logic.eigenfaces import PCASVM
from app.logic.face_detectors.hog_face import HogFaceDetector
from app.logic.utils.get_dataset import split_img
import cv2 as cv
import os
from app.utils.constants import ML_MODEL_EXT, MODEL_DIR
from app.logic.model_table import ModelTable


if __name__ == '__main__':

    # model = PCASVM(['MinhDoan', 'HuyDao'], k=25)
    # acc = model.fit()

    # path = os.path.join(MODEL_DIR, 'pcasvm_test' + ML_MODEL_EXT)

    # model.save(path)

    table = ModelTable()
    # table.add_model('pcasvm_test', 'ml', path, acc)
    table.load_table()

    path = table.get_model('pcasvm_test', return_path=True)

    blank_model = PCASVM()
    blank_model.load(path)

    capture = cv.VideoCapture(0)

    detector = HogFaceDetector()

    while True:
        _, frame = capture.read()
        pred_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect face
        faces = detector.detect_faces(pred_frame)

        for face in faces:
            cord = detector.get_cordinate(face)
            x, y, w, h = cord
            # Resize face
            face_resized = split_img(pred_frame, cord)
            # Predict
            pred = blank_model.predict(face_resized)
            # Show face
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, pred, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv.LINE_AA)

        # Show frame
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
