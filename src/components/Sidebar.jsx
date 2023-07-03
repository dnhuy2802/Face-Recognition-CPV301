import { useContext } from "react";
import { Menu } from "antd";
import { StateContext } from "../contexts/stateContext";
import { sidebarOptions } from "../utils/sidebarOptions";

function SideBar() {
  const { store } = useContext(StateContext);
  const _defaultSelectedKeys = store((state) => state.selectedSidebarItem);
  const _setter = store((state) => state.setSelectedSidebarItem);

  function onMenuItemClick({ key }) {
    _setter(key);
  }

  return (
    <Menu
      items={sidebarOptions}
      defaultSelectedKeys={_defaultSelectedKeys}
      onClick={onMenuItemClick}
    />
  );
}

export default SideBar;
