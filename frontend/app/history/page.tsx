import Link from "next/link";

import { StatusBadge } from "@/components/status-badge";
import { getHistory } from "@/lib/api";

export default async function HistoryPage() {
  const history = await getHistory();

  return (
    <main className="space-y-6">
      <div className="flex justify-between">
        <h1 className="font-display text-4xl">Verification History</h1>
        <Link href="/">Back home</Link>
      </div>
      <div className="panel overflow-hidden">
        <div className="divide-y divide-black/10">
          {history.items.map((item) => (
            <Link
              key={item.job_id}
              href={`/results/${item.job_id}`}
              className="flex flex-col gap-3 p-4 transition hover:bg-black/[0.03] md:flex-row md:items-center md:justify-between"
            >
              <div>
                <div className="font-medium">{item.headline_message}</div>
                <div className="text-sm text-black/60">
                  {item.input_type} • {new Date(item.created_at).toLocaleString()}
                </div>
              </div>
              <StatusBadge status={item.status} />
            </Link>
          ))}
        </div>
      </div>
    </main>
  );
}

