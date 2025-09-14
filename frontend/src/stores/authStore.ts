import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface AuthState {
  isLoggedIn: boolean;
  login: () => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      isLoggedIn: false,
      login: () => set({ isLoggedIn: true }),
      logout: () => set({ isLoggedIn: false }),
    }),
    {
      name: "auth-storage",
    }
  )
);
