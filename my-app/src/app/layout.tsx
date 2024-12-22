// src/app/layout.tsx

import React from "react";
import "./globals.css";

export const metadata = {
  title: "trendscan by talialabs",
  description: "an open-source tools from the 'talialabs' brand. see the latest top 50 trends in the fashion industry in just a click.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Global elements like headers or navigation can go here */}
        <main>{children}</main> {/* The page content will be rendered here */}
      </body>
    </html>
  );
}
