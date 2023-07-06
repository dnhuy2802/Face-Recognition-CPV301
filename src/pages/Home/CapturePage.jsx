import style from "./CapturePage.module.css";
import { useMemo, useState, useContext, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Camera from "../../components/Camera";
import { useEffect } from "react";
import { Button } from "antd";
import { CloseOutlined } from "@ant-design/icons";
import { StateContext } from "../../contexts/stateContext";
import { appStrings } from "../../utils/appStrings";
import { CAPTURE_AMOUNT } from "../../utils/constants";

let captureImageCallback = null;
let interval = null;

function CapturePage() {
  const stepAngle = (2 * Math.PI) / CAPTURE_AMOUNT;
  const circleRadius = 200;

  const { store } = useContext(StateContext);
  const setGlobalImages = store((state) => state.setImages);
  const [images, setImages] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);

  const navigator = useNavigate();
  const location = useLocation();

  // Get name of person from UploadScreen
  const name = location.state?.name;

  // Handle close button
  function handleCloseButton() {
    navigator("/");
  }

  // Get screenshot callback
  const mediaCallback = useMemo(
    () =>
      ({ getScreenshot }) => {
        captureImageCallback = getScreenshot;
      },
    []
  );

  // Add image to state
  const addImage = useCallback((image) => {
    setImages((prev) => [...prev, image]);
  }, []);

  // Capture image
  function onCapturePressed() {
    setIsCapturing(true);
    interval = setInterval(() => {
      const image = captureImageCallback();
      addImage(image);
    }, 200);
  }

  // If name is not set, redirect to home page
  useEffect(() => {
    if (!name) {
      navigator("/");
    }
  }, []);

  // Stop capturing
  useEffect(() => {
    if (images.length === CAPTURE_AMOUNT) {
      setIsCapturing(false);
      clearInterval(interval);
      setGlobalImages(images);
      navigator("/");
    }
  }, [images]);

  return (
    <div className={style.container}>
      <div className={style.cameraContainer}>
        <Camera showSelectDevice={false} mediaCallback={mediaCallback} />
        <div className={style.blurOverlay}></div>
        <div className={style.indicatorOverlay}>
          {[...Array(CAPTURE_AMOUNT)].map((_, index) => {
            const currentAngle = index * stepAngle;
            const x_pos = Math.floor(Math.sin(currentAngle) * circleRadius);
            const y_pos = Math.floor(Math.cos(currentAngle) * circleRadius);
            const indicatorElementStyle = {
              inset: `50% auto auto 50%`,
              transform: `translate(${x_pos}px, ${y_pos}px)`,
              backgroundColor:
                index <= images.length - 1
                  ? "var(--status-success)"
                  : "var(--color-secondary)",
            };
            return (
              <div
                key={index}
                className={style.indicatorElement}
                style={indicatorElementStyle}
              ></div>
            );
          })}
        </div>
      </div>
      <div className={style.nameOverlay}>{name}</div>
      <div className={style.closeButton}>
        <Button
          icon={<CloseOutlined />}
          type="primary"
          onClick={handleCloseButton}
        />
      </div>
      <div className={style.captureButtonContainer}>
        <Button
          type="primary"
          onClick={onCapturePressed}
          className={style.captureButton}
          loading={isCapturing}
        >
          {appStrings.upload.captureButton}
        </Button>
      </div>
    </div>
  );
}

export default CapturePage;
