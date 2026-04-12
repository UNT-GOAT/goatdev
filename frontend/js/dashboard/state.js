(function (global) {
  const CAMERA_COLD_MESSAGE =
    "Cameras are below the minimum operating temperature. Please wait for the heating system.";

  const dashboard = (global.HerdDashboard = global.HerdDashboard || {});
  const state = (dashboard.state = {
    allAnimals: [],
    allProviders: [],
    allGrades: [],
    providerMap: {},
    loadErrors: [],
    animalFilter: "all",
    animalSort: "id-desc",
    provFilter: "all",
    selectedAnimalId: null,
    selectedAnimalType: null,
    currentPage: "dashboard",
    pendingGradeResult: null,
    _gradeAnimalCreated: false,
    _gradeExistingId: null,
    _gradeReservedId: null,
    _reviewGradeId: null,
    _reviewCurrentResult: null,
    _gradeAnnotationContext: null,
    _editAnimalGradeState: null,
    _reviewEditGradeState: null,
    nextGradeId: null,
    piConnected: false,
    cameraColdBlocked: false,
    cameraColdMessage: CAMERA_COLD_MESSAGE,
    cameraSafety: null,
    _currentUser: null,
    _cachedNextId: null,
    _inlineProviderSelect: null,
    camerasActive: false,
    _devForceSave: false,
    _auditLogs: [],
  });

  [
    "allAnimals",
    "allProviders",
    "allGrades",
    "providerMap",
    "loadErrors",
    "animalFilter",
    "animalSort",
    "provFilter",
    "selectedAnimalId",
    "selectedAnimalType",
    "currentPage",
    "pendingGradeResult",
    "_gradeAnimalCreated",
    "_gradeExistingId",
    "_gradeReservedId",
    "_reviewGradeId",
    "_reviewCurrentResult",
    "_gradeAnnotationContext",
    "_editAnimalGradeState",
    "_reviewEditGradeState",
    "nextGradeId",
    "piConnected",
    "cameraColdBlocked",
    "cameraColdMessage",
    "cameraSafety",
    "_currentUser",
    "_cachedNextId",
    "_inlineProviderSelect",
    "camerasActive",
    "_devForceSave",
    "_auditLogs",
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
