import { create } from "zustand";

function useProviderState(states) {
  /// Map each state to a setter
  function getStateSetter(setFunc) {
    const stateSetter = Object.keys(states).reduce((acc, key) => {
      const setterKey = `set${key.charAt(0).toUpperCase() + key.slice(1)}`;
      acc[setterKey] = (value) => setFunc({ [key]: value });
      return acc;
    }, {});
    return stateSetter;
  }

  const store = create((set) => ({
    ...states,
    ...getStateSetter(set),
  }));

  return store;
}

export default useProviderState;
