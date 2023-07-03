import style from "./UploadScreen.module.css";
import { useState, useMemo, useRef } from "react";
import { Space, Button, Typography, Modal, Input, Empty } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import { AiFillCamera } from "react-icons/ai";
import { appStrings } from "../../../utils/appStrings";
import Spacer from "../../../components/Spacer";
import Grid from "../../../components/Grid";
import FaceCard from "../../../components/FaceCard";
import Camera from "../../../components/Camera";
import { MOCK_DATA } from "../../../utils/mockData";
import useProviderState from "../../../hooks/useProviderState";
import { initialState } from "../../../contexts/nameInputContext";
import Flex from "../../../components/Flex";
import { convertLowerCase, trimString } from "../../../utils/utilities";
import DebouncedInput from "../../../components/DebouncedInput";

function UploadScreen() {
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
    if (MOCK_DATA.find((item) => item.name === _inputValue)) {
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
      toggleModal();
    }
  }

  // Get face cards
  function getFaceCards() {
    // Filter data based on search value
    const _data = MOCK_DATA.filter((item) =>
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
              imgUrl={item.imgUrl}
              name={item.name}
            />
          ))}
        </Grid>
      );
    }
  }

  return (
    <>
      <Space direction="vertical" block className={style.container}>
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
    </>
  );
}

export default UploadScreen;
