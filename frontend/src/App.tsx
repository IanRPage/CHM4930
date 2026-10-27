import { Route, Routes } from "react-router-dom";
import Layout from "./components/layout/Layout.tsx";
import AboutPage from "./pages/AboutPage.tsx";
import BatchPredictionPage from "./pages/BatchPredictionPage.tsx";
import DatasetExplorerPage from "./pages/DatasetExplorerPage.tsx";
import HomePage from "./pages/HomePage.tsx";
import SinglePredictionPage from "./pages/SinglePredictionPage.tsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="predict" element={<SinglePredictionPage />} />
        <Route path="batch" element={<BatchPredictionPage />} />
        <Route path="explorer" element={<DatasetExplorerPage />} />
        <Route path="about" element={<AboutPage />} />
      </Route>
    </Routes>
  );
}
