(function (global) {
  const setup = (global.HerdSetup = global.HerdSetup || {});
  const state = (setup.state = {
    _currentUser: null,
    currentPage: "viewfocus",
    piConnected: false,
    focusValues: { side: 200, top: 200, front: 200 },
    focusStreamsActive: false,
    captureCamsActive: false,
    _focusThrottleTimers: {},
    _afActive: { side: false, top: false, front: false },
    _heaterInterval: null,
    _heaterHistoryInterval: null,
    captureGoatId: 1,
    isCapturing: false,
  });

  [
    "_currentUser",
    "currentPage",
    "piConnected",
    "focusValues",
    "focusStreamsActive",
    "captureCamsActive",
    "_focusThrottleTimers",
    "_afActive",
    "_heaterInterval",
    "_heaterHistoryInterval",
    "captureGoatId",
    "isCapturing",
  ].forEach((key) => {
    Object.defineProperty(global, key, {
      configurable: true,
      enumerable: false,
      get() {
        return state[key];
      },
      set(value) {
        state[key] = value;
      },
    });
  });
})(window);
