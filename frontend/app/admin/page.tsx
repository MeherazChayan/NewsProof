import Link from "next/link";

import { PublisherAdmin } from "@/components/publisher-admin";

export default function AdminPage() {
  return (
    <main className="space-y-6">
      <div className="flex justify-between">
        <h1 className="font-display text-4xl">Trusted Publisher Registry</h1>
        <Link href="/">Back home</Link>
      </div>
      <PublisherAdmin />
    </main>
  );
}
