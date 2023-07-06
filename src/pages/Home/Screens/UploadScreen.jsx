import style from "./UploadScreen.module.css";
import { useState, useMemo, useRef, useContext, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Space, Button, Typography, Modal, Input, Empty, Progress } from "antd";
import { SearchOutlined, CloudUploadOutlined } from "@ant-design/icons";
import { AiFillCamera } from "react-icons/ai";
import { appStrings } from "../../../utils/appStrings";
import Spacer from "../../../components/Spacer";
import Grid from "../../../components/Grid";
import FaceCard from "../../../components/FaceCard";
import Camera from "../../../components/Camera";
import useProviderState from "../../../hooks/useProviderState";
import { initialState, InputContext } from "../../../contexts/nameInputContext";
import Flex from "../../../components/Flex";
import { convertLowerCase, trimString } from "../../../utils/utilities";
import DebouncedInput from "../../../components/DebouncedInput";
import { StateContext } from "../../../contexts/stateContext";
import {
  uploadFaces,
  getFaces,
  deleteFace,
} from "../../../apis/registerFacesService";
import { BASE64_PREFIX } from "../../../utils/constants";

function UploadScreen() {
  // Navigation
  const navigator = useNavigate();

  // Searching
  const [searchValue, setSearchValue] = useState("");

  // Name input
  const store = useMemo(() => useProviderState(initialState), []);
  const inputRef = useRef(null);
  const _value = store((state) => state.value);
  const _setValue = store((state) => state.setValue);
  const _error = store((state) => state.error);
  const _setError = store((state) => state.setError);
  const _errorMessage = store((state) => state.errorMessage);
  const _setErrorMessage = store((state) => state.setErrorMessage);

  // Modal
  const [isModalVisible, setIsModalVisible] = useState(false);

  // Images
  const { store: globalStore } = useContext(StateContext);
  const images = globalStore((state) => state.images);
  const setImages = globalStore((state) => state.setImages);
  const uploadName = globalStore((state) => state.uploadName);
  const setUploadName = globalStore((state) => state.setUploadName);

  // Faces
  const [faces, setFaces] = useState([]);

  // Toggle modal handler
  const toggleModal = () => setIsModalVisible((prev) => !prev);

  // Validate input value
  function _validateInputValue(inputValue) {
    const _inputValue = trimString(inputValue);
    // Check if input is empty
    if (!_inputValue.length > 0) {
      _setError(true);
      _setErrorMessage(appStrings.upload.modalNameInputEmptyError);
      return;
    }
    // Check if input is duplicate
    if (faces.find((item) => item.name === _inputValue)) {
      _setError(true);
      _setErrorMessage(appStrings.upload.modalNameInputDuplicateError);
      return;
    }
    // Reset error
    _setError(false);
    _setErrorMessage("");
  }

  // Name input value change handler
  function onInputValueChange(e) {
    const _inputValue = e.target.value;
    _validateInputValue(_inputValue);
    _setValue(_inputValue);
  }

  // Start capture handler
  function onStartCapture() {
    _validateInputValue(_value);
    // If error or no value
    if (_error || !_value) {
      inputRef.current.focus();
    } else {
      // Navigate to capture screen
      navigator("/capture", { state: { name: _value } });
      // Set upload name
      setUploadName(_value);
      toggleModal();
    }
  }

  function onDeleteFace(name) {
    deleteFace(name).then(() => {
      getFaces().then((res) => setFaces(res));
    });
  }

  useEffect(() => {
    // If have images, start upload
    if (images.length > 0) {
      uploadFaces(uploadName, images).then(() => setImages([]));
    }
    // Fetch faces
    getFaces().then((res) => setFaces(res));
  }, [images]);

  // Get Upload Content. If have images, start upload
  function getUploadContent() {
    if (images.length === 0) {
      return (
        <Button
          type="dashed"
          className={style.captureButton}
          onClick={toggleModal}
        >
          <Space direction="vertical" size={1}>
            <AiFillCamera size={24} />
            {appStrings.upload.captureButton}
          </Space>
        </Button>
      );
    } else {
      return (
        <Button type="dashed" className={style.captureButton}>
          <Flex direction="column" align="center">
            <Space direction="horizontal" size={10}>
              <CloudUploadOutlined />
              {appStrings.upload.uploadingTitle}
              <Typography.Text type="secondary">60%</Typography.Text>
            </Space>
            <Progress
              percent={60}
              status="active"
              showInfo={false}
              size="small"
              strokeColor={{
                "0%": "var(--color-secondary)",
                "100%": "var(--color-primary)",
              }}
              className={style.uploadProgress}
            />
          </Flex>
        </Button>
      );
    }
  }

  // Get face cards
  function getFaceCards() {
    // Filter data based on search value
    const _data = faces.filter((item) =>
      convertLowerCase(item.name).includes(
        trimString(convertLowerCase(searchValue))
      )
    );
    // If no data found
    if (_data.length === 0) {
      return (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={appStrings.upload.noDataFound}
        />
      );
    } else {
      // If data found
      return (
        <Grid>
          {_data.map((item) => (
            <FaceCard
              key={item.id}
              id={item.id}
              imgUrl={`${BASE64_PREFIX}${item.thumbnail}`}
              name={item.name}
              onDelete={() => onDeleteFace(item.name)}
            />
          ))}
        </Grid>
      );
    }
  }

  return (
    <InputContext.Provider value={{ store: store }}>
      <Space direction="vertical" block className={style.container}>
        {getUploadContent()}
        <Spacer size={10} />
        <Flex justify="space-between" direction="row">
          <div>
            <Typography.Title level={5}>
              {appStrings.upload.registeredTitle}
            </Typography.Title>
          </div>
          <DebouncedInput
            onChange={setSearchValue}
            size="small"
            className={style.searchInput}
            prefix={<SearchOutlined />}
            placeholder={appStrings.upload.searchPlaceholder}
            allowClear
          />
        </Flex>
        {getFaceCards()}
      </Space>
      <Modal
        open={isModalVisible}
        onCancel={toggleModal}
        width={"70%"}
        footer={
          <Button
            type="primary"
            className={style.startCaptureButton}
            onClick={onStartCapture}
          >
            {appStrings.upload.modalStartCaptureButton}
          </Button>
        }
      >
        <Typography.Title level={5}>
          {appStrings.upload.modalNameInputTitle}
        </Typography.Title>
        <Input
          ref={inputRef}
          placeholder={appStrings.upload.modalNameInputPlaceholder}
          value={_value}
          onChange={onInputValueChange}
          status={_error ? "error" : "validating"}
        />
        {_error && (
          <Typography.Text type="danger">{_errorMessage}</Typography.Text>
        )}
        <Spacer size={10} />
        <Camera />
      </Modal>
    </InputContext.Provider>
  );
}

export default UploadScreen;
