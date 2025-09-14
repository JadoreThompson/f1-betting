import { ArrowTrendingUpIcon } from "@heroicons/react/24/outline";
import { useEffect, useState, type ComponentType } from "react";
import { useNavigate } from "react-router";
import { UtilsManager } from "../lib/classes/UtilsManager";
import { useAuthStore } from "../stores/authStore";

interface LoginProtectedProps<T extends object> {
  Component: ComponentType<T>;
  args?: T;
}

export function LoginProtected<T extends object>({
  Component,
  args,
}: LoginProtectedProps<T>) {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const { login, logout } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    const verifyUser = async (): Promise<void> => {
      try {
        const rsp = await fetch(UtilsManager.BASE_URL + "/auth/me", {
          credentials: "include",
        });

        if (!rsp.ok) throw new Error(`Auth failed: ${rsp.status}`);

        login();
        setIsLoading(false);
      } catch (error) {
        logout();
        navigate("/login");
      }
    };

    verifyUser();
  }, [login, logout, navigate]);

  if (isLoading) {
    return (
      <div className="w-full h-screen fixed top-0 left-0 flex-center bg-white/10 backdrop-blur-sm">
        <ArrowTrendingUpIcon height={30} width={30} />
      </div>
    );
  }

  return args ? <Component {...args} /> : <Component {...({} as T)} />;
}
