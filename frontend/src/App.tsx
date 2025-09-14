import { BrowserRouter, Route, Routes } from "react-router";

import PredictionsPage from "./pages/PredictionsPage";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PredictionsPage />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
