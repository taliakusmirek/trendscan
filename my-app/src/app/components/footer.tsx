import Image from 'next/image';
import Link from 'next/link';

export default function Footer() {
    return (
        <div className='flex justify-center space-x-10 text-brand-Purple mt-28'>
            <a href="https://talialabs.org" target="_blank" className="text-base-mobile sm:text-base-desktop"><h4 className="mt-4">made by talialabs</h4></a>
        </div>
    );
 }
