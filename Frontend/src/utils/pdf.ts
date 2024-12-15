export const isPDFUrl = (url: string): boolean => {
  return url.toLowerCase().endsWith('.pdf');
};

export const getSafePDFUrl = (url: string): string => {
  // List of trusted domains for PDFs
  const trustedDomains = [
    'www.w3.org',
    'www.mouser.com',
    'ww1.microchip.com',
    'www.ti.com'
  ];

  try {
    const urlObj = new URL(url);
    if (trustedDomains.includes(urlObj.hostname)) {
      return url;
    }
    // Fallback to a default PDF if the domain isn't trusted
    return 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf';
  } catch {
    return 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf';
  }
};