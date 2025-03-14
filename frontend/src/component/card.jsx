// src/components/ui/card.jsx
import React from "react";

export const Card = ({ children }) => {
  return (
    <div className="bg-white shadow-md rounded-lg p-6 w-full max-w-md">
      {children}
    </div>
  );
};

export const CardContent = ({ children }) => {
  return <div className="mt-4">{children}</div>;
};
