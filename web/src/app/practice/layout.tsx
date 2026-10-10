import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'TorchCode · Your daily spark',
  description: 'Build your PyTorch skills, one joyful practice session at a time.',
};

export default function PracticeLayout({ children }: { children: React.ReactNode }) {
  return children;
}
