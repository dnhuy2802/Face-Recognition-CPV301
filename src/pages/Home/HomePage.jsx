import style from "./HomePage.module.css";
import { useContext } from "react";
import { Layout } from "antd";
import { appStrings } from "../../utils/appStrings";
import SideBar from "../../components/Sidebar";
import { BsPersonBoundingBox } from "react-icons/bs";
import Spacer from "../../components/Spacer";
import { sidebarOptions } from "../../utils/sidebarOptions";
import { StateContext } from "../../contexts/stateContext";

const { Header, Footer, Sider, Content } = Layout;

function HomePage() {
  const { store } = useContext(StateContext);
  const _currentSelectedScreen = store((state) => state.selectedSidebarItem);

  function _getScreen() {
    const _selectedScreen = sidebarOptions.find(
      (item) => item.key === _currentSelectedScreen
    );
    return _selectedScreen?.screen;
  }

  return (
    <Layout className={style.wrapper}>
      <Header className={style.headerTitle}>
        <BsPersonBoundingBox />
        <Spacer size={10} />
        {appStrings.appName}
      </Header>
      <Layout hasSider className={style.container}>
        <Sider className={style.sidebarContainer}>
          <SideBar />
        </Sider>
        <Content className={style.content}>{_getScreen()}</Content>
      </Layout>
    </Layout>
  );
}

export default HomePage;
