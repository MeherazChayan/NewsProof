import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "NewsProof",
  description: "Cryptographic news provenance verification"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="mx-auto min-h-screen max-w-6xl px-4 py-6 md:px-6">{children}</div>
      </body>
    </html>
  );
}

