from app.logic.ml.knn import KNN
from app.logic.face_detectors.haar_cascade import HaarCascadeDetector
import cv2 as cv
from app.utils.constants import TRAINING_IMAGE_SIZE


if __name__ == '__main__':

    model = KNN(['MinhDoan', 'HuyDao'], k=25)
    model.fit()

    capture = cv.VideoCapture(0)

    detector = HaarCascadeDetector()

    while True:
        _, frame = capture.read()
        pred_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect face
        faces = detector.detect_faces(pred_frame)

        for face in faces:
            (x, y, w, h) = detector.get_cordinate(face)
            # Resize face
            face_resized = pred_frame[y:y+h, x:x+w]
            # Predict
            pred = model.predict(face_resized)
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
